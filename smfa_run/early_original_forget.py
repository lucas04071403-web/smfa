"""Judge completed Original questions while the remaining generation runs."""

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "smfa_run"
sys.path.insert(0, str(ROOT / "reproduce_smfa"))
import run as runner
import score


def input_snapshot(folder, split, profiles):
    if profiles is None:
        data = (folder / f"{split}_generation_results.json").read_bytes()
        return json.loads(data)["Generation_Questions"], data
    questions = []
    for index in range(profiles):
        path = folder / ".profiles" / split / f"{index:04d}" / f"{split}_generation_results.json"
        questions.extend(json.loads(path.read_text())["Generation_Questions"])
    return questions, (json.dumps({"Generation_Questions": questions}, indent=4) + "\n").encode()


def verified_cache(path, questions, field):
    records = {}
    if not path.exists():
        return records
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise RuntimeError(f"Cache ends with an incomplete record: {path}")
    for line in data.splitlines():
        row = json.loads(line)
        index = row["index"]
        if type(index) is not int or index < 0:
            raise RuntimeError(f"Invalid cache index: {path}")
        if index >= len(questions):
            continue
        question = questions[index]
        if row["digest"] != score.digest(question):
            raise RuntimeError(f"Cache refers to a different generation: {path}, {index}")
        result = row["result"]
        if type(result.get(field)) is not int or not 0 <= result[field] <= 10:
            continue
        if not all(result.get(dest) == question.get(src) for dest, src in (
            ("Question", "question"), ("Generated Answer", "generated_answer"),
            ("Ground Truth", "ground_truth"), ("Image_ID", "image_id"),
            ("image_type", "image_type"), ("Question Type", "question type"),
        )):
            raise RuntimeError(f"Cache score and generation differ: {path}, {index}")
        if not isinstance(result.get("Justification"), str):
            raise RuntimeError(f"Cache justification is absent: {path}, {index}")
        records[index] = row
    return records


def append_verified_cache(source, destination, questions, field):
    records = verified_cache(source, questions, field)
    if set(records) != set(range(len(questions))):
        raise RuntimeError(f"Completed scoring cache has missing scores: {source}")
    existing = verified_cache(destination, questions, field)
    for index in records.keys() & existing.keys():
        if records[index] != existing[index]:
            raise RuntimeError(f"Conflicting cached score at index {index}")
    additions = [records[index] for index in sorted(records.keys() - existing.keys())]
    if additions:
        with destination.open("ab") as handle:
            handle.write(b"".join((json.dumps(row) + "\n").encode() for row in additions))
            handle.flush()
            os.fsync(handle.fileno())
    if verified_cache(destination, questions, field) != records:
        raise RuntimeError("Published score cache differs from validated scores")
    return len(additions)


def require_generation_running(split, target):
    status = json.loads((RUN / "parallel_original_status.json").read_text())
    if status["state"] != "running" or status["stage"] != "evaluate_Original":
        raise RuntimeError("Full Original scoring may have started; keep early results isolated")
    os.kill(status["child_pid"], 0)
    if (target / f"{split}_generation_results.json").exists():
        raise RuntimeError("Full split generation finished; keep early results isolated")
    progress = json.loads((target / "progress.json").read_text())
    if progress["split"] != split or progress["completed_profiles"] >= progress["total_profiles"] - 1:
        raise RuntimeError("Generation is near its scoring handoff; keep early results isolated")


def main(args, label):
    work = RUN / label
    status_path = RUN / f"{label}_status.json"
    with (RUN / f"{label}.lock").open("w") as lock, (RUN / "early_original.lock").open("w") as shared_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(shared_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((RUN / "manifest.json").read_text())
        for relative, expected in manifest["source_sha256"].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Experiment source changed: {relative}")
        original = json.loads((RUN / "parallel_original_status.json").read_text())
        source_folder = Path(original["staging"])
        if not source_folder.exists():
            source_folder = Path(original["destination"])
        questions, snapshot = input_snapshot(source_folder, args.split, args.profiles)
        import pandas as pd
        from evaluate import verify_questions
        rows = pd.read_parquet(runner.REPO / f"data/S-MLLMUn-data/eval_data/{args.split}_5/train-00000-of-00001.parquet")
        total_profiles = len(rows)
        if args.profiles is not None:
            if not 0 < args.profiles <= total_profiles:
                raise ValueError("Requested profile prefix is outside the evaluation dataset")
            rows = rows.iloc[:args.profiles]
        verify_questions(questions, rows)
        index = 0
        for _, row in rows.iterrows():
            for kind, column in (("image_understanding", "image understanding"),
                                 ("image_memory", "image memory"), ("text_memory", "text memory")):
                for source_question in row[column]:
                    question = questions[index]
                    if kind != "text_memory":
                        assert question["image_id"] == row["ID"]
                        assert question["image_type"] == source_question["Image"]
                    index += 1
        work.mkdir(exist_ok=True)
        local_input = work / f"{args.split}_generation_results.json"
        if local_input.exists():
            if local_input.read_bytes() != snapshot:
                raise RuntimeError("Original generation snapshot changed")
        else:
            local_input.write_bytes(snapshot)
        state = {
            "pid": os.getpid(), "state": "running", "started_at": runner.now(),
            "input_sha256": hashlib.sha256(snapshot).hexdigest(),
            "source_sha256": manifest["source_sha256"], "workers": 4,
            "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "questions": len(questions), "completed_metrics": [],
            "split": args.split, "profiles": len(rows), "total_split_profiles": total_profiles,
            "partial_split": len(rows) != total_profiles,
            "scope": f"Original {args.split}: {len(rows)}/{total_profiles} profiles; complete experiment results remain pending",
            "reused_scores": {},
        }
        runner.write_json(status_path, state)
        try:
            for name, field, suffix in (
                ("eval_fact", "Factuality Score", "factuality"),
                ("eval_meaningful", "Meaningful Score", "meaningful"),
            ):
                state["stage"] = suffix
                runner.write_json(status_path, state)
                module = importlib.import_module(name)
                if module.client is None:
                    raise RuntimeError("Judge credential is absent")
                module.client = module.client.with_options(timeout=120, max_retries=2)
                cache_name = f"{args.split}_generation_results_{suffix}.jsonl"
                local_cache = work / cache_name
                reusable = verified_cache(source_folder / cache_name, questions, field)
                if not local_cache.exists() and reusable:
                    local_cache.write_text("".join(json.dumps(reusable[index]) + "\n" for index in sorted(reusable)))
                state["reused_scores"][suffix] = len(verified_cache(local_cache, questions, field))
                runner.write_json(status_path, state)
                score.score_file(module, local_input, field, suffix, 4)
                output = json.loads((work / f"{args.split}_generation_results_{suffix}_score.json").read_text())
                counts = Counter(q["question type"] for q in questions)
                for kind, count in counts.items():
                    assert len(output[kind]) == count, f"Incomplete {suffix}: {kind}"
                    for question, result in zip((q for q in questions if q["question type"] == kind), output[kind]):
                        assert type(result[field]) is int and 0 <= result[field] <= 10
                        assert all(result[dest] == question.get(src) for dest, src in (
                            ("Question", "question"), ("Generated Answer", "generated_answer"),
                            ("Ground Truth", "ground_truth"), ("Image_ID", "image_id"),
                        )), "Score and generation differ"
                state["completed_metrics"].append(suffix)
                state[f"{suffix}_means"] = output["Score"]
                runner.write_json(status_path, state)
            current = json.loads((RUN / "parallel_original_status.json").read_text())
            target = Path(current["staging"])
            if not target.exists():
                target = Path(current["destination"])
            _, current_snapshot = input_snapshot(target, args.split, args.profiles)
            if current_snapshot != snapshot:
                raise RuntimeError("Destination generation differs from judged input")
            promoted = []
            existing = []
            appended = {}
            for suffix, field in (("factuality", "Factuality Score"), ("meaningful", "Meaningful Score")):
                names = [f"{args.split}_generation_results_{suffix}.jsonl"]
                if len(rows) == total_profiles:
                    names.append(f"{args.split}_generation_results_{suffix}_score.json")
                for name in names:
                    # Linking a closed file publishes it atomically without replacing a writer.
                    try:
                        os.link(work / name, target / name)
                        promoted.append(name)
                    except FileExistsError:
                        existing.append(name)
                        if name.endswith(".jsonl") and not os.path.samefile(work / name, target / name):
                            require_generation_running(args.split, target)
                            appended[name] = append_verified_cache(work / name, target / name, questions, field)
            state.update(state="complete", stage="complete", completed_at=runner.now(),
                         promoted=promoted, existing_not_replaced=existing, appended_scores=appended)
            runner.write_json(status_path, state)
            print(f"Verified {len(questions)} Original {args.split} questions for both metrics; partial_split={state['partial_split']}", flush=True)
        except BaseException as error:
            state.update(state="failed", failed_at=runner.now(), error=str(error))
            runner.write_json(status_path, state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--split", choices=("forget", "retain"), default="forget")
    parser.add_argument("--profiles", type=int)
    args = parser.parse_args()
    if args.profiles is not None and args.profiles <= 0:
        parser.error("--profiles must be positive")
    label = f"early_original_{args.split}"
    if args.profiles is not None:
        label += f"_{args.profiles}"
    if args.launch:
        status = json.loads((RUN / "parallel_original_status.json").read_text())
        if status["state"] != "running" or status["stage"] != "evaluate_Original":
            raise RuntimeError("Original generation must still be running before early judging")
        os.kill(status["child_pid"], 0)
        raw = Path(f"/proc/{status['pid']}/environ").read_bytes()
        environment = {os.fsdecode(k): os.fsdecode(v) for k, v in
                       (part.split(b"=", 1) for part in raw.split(b"\0") if b"=" in part)}
        environment["CUDA_VISIBLE_DEVICES"] = ""
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--split", args.split]
        if args.profiles is not None:
            command.extend(["--profiles", str(args.profiles)])
        with (RUN / f"{label}.log").open("a") as log:
            child = subprocess.Popen(command,
                                     env=environment, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Launched {label} judging PID {child.pid}")
    else:
        main(args, label)
