"""Judge complete SMFA profile prefixes while its generation continues."""

import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "smfa_run"
REPO = ROOT / "S-MLLMUn"
FOLDER = REPO / "eval_result/llava-onevision/split_ratio_5/SMFA"
STATUS = RUN / "early_smfa_queue_status.json"
TARGETS = (("forget", 50),) + tuple(("retain", n) for n in (125, 250, 375, 500, 625, 750))
HELPER_HASH = "72eb6db8117c7b62788ed7a21d9c4f1a07dbaae38426d61b12063f138a2315fd"
AUDITOR_HASH = "d26f4f9ed67b85c1980df7434cf32f1c4ea144723d8fc487020cd00fddfbbd44"
sys.path.insert(0, str(ROOT / "reproduce_smfa"))
import run as runner
import score


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def live(pid):
    try:
        return not Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return False


def label(split, profiles):
    return f"early_smfa_{split}_{profiles}"


def verify_sources(expected=None):
    if expected and sha(Path(__file__)) != expected:
        raise RuntimeError("SMFA early judge changed after queue launch")
    manifest = read(RUN / "manifest.json")
    if sys.prefix != manifest["conda_prefix"]:
        raise RuntimeError("Activate the recorded Conda environment")
    for relative, expected_hash in manifest["source_sha256"].items():
        if sha(ROOT / relative) != expected_hash:
            raise RuntimeError(f"Experiment source changed: {relative}")
    if sha(RUN / "early_original_forget.py") != HELPER_HASH:
        raise RuntimeError("The verified early-cache helper changed")
    if sha(RUN / "audit_final.py") != AUDITOR_HASH:
        raise RuntimeError("The independent auditor changed")
    return manifest


def generation_progress():
    status = read(RUN / "status.json")
    if status["state"] == "failed":
        raise RuntimeError("Main pipeline failed; keep early SMFA results isolated")
    if status["state"] == "complete" or status["stage"] in (
        "score_SMFA", "evaluate_Original", "score_Original", "complete"
    ):
        return None
    if status["stage"] != "evaluate_SMFA":
        if not live(status["pid"]):
            raise RuntimeError("Main pipeline stopped before SMFA generation")
        return {"phase": "waiting_for_smfa_generation"}
    if not live(status["pid"]):
        raise RuntimeError("Main pipeline stopped during SMFA generation")
    if "child_pid" not in status or not live(status["child_pid"]):
        return {"phase": "waiting_for_generator_stage_update"}
    if (FOLDER / "retain_generation_results.json").exists():
        return None
    original = read(RUN / "parallel_original_status.json")
    if original["state"] != "complete":
        return {"phase": "waiting_for_original_judging"}
    try:
        progress = read(FOLDER / "progress.json")
    except (FileNotFoundError, json.JSONDecodeError):
        return {"phase": "waiting_for_profile_progress"}
    if progress["split"] == "retain" and progress["completed_profiles"] >= progress["total_profiles"] - 2:
        return None
    return dict(progress, phase="generating")


def ready(progress, split, profiles):
    if not progress or progress["phase"] != "generating":
        return False
    if split == "forget":
        return (FOLDER / "forget_generation_results.json").exists()
    return progress["split"] == "retain" and progress["completed_profiles"] >= profiles


def worker(args):
    import pandas as pd
    import early_original_forget as helper
    import audit_final as auditor

    name = label(args.split, args.profiles)
    state_path = RUN / f"{name}_status.json"
    with (RUN / "early_smfa.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify_sources(args.expected_sha)
        if not ready(generation_progress(), args.split, args.profiles):
            raise RuntimeError("SMFA prefix is unavailable or the complete scorer may have started")
        questions, snapshot = helper.input_snapshot(FOLDER, args.split, args.profiles)
        rows = pd.read_parquet(REPO / f"data/S-MLLMUn-data/eval_data/{args.split}_5/train-00000-of-00001.parquet")
        total_profiles = len(rows)
        if not 0 < args.profiles <= total_profiles:
            raise ValueError("Invalid profile prefix")
        rows = rows.iloc[:args.profiles]
        from evaluate import verify_questions
        verify_questions(questions, rows)
        expected = [(kind, row, item) for _, row in rows.iterrows()
                    for kind, column in zip(auditor.KINDS, auditor.COLUMNS) for item in row[column]]
        for question, (kind, row, item) in zip(questions, expected):
            if not isinstance(question.get("generated_answer"), str) or not question["generated_answer"].strip():
                raise RuntimeError("Empty SMFA generation")
            if kind != "text_memory" and (question.get("image_id"), question.get("image_type")) != (row["ID"], item["Image"]):
                raise RuntimeError("SMFA image identity differs")
        work = RUN / name
        work.mkdir(exist_ok=True)
        local_input = work / f"{args.split}_generation_results.json"
        if local_input.exists() and local_input.read_bytes() != snapshot:
            raise RuntimeError("SMFA prefix generation changed")
        if not local_input.exists():
            local_input.write_bytes(snapshot)
        state = {"state": "running", "pid": os.getpid(), "started_at": runner.now(),
                 "split": args.split, "profiles": args.profiles, "questions": len(questions),
                 "total_split_profiles": total_profiles, "workers": 4, "completed_metrics": [],
                 "input_sha256": sha(local_input), "launcher_sha256": sha(Path(__file__)),
                 "source_sha256": manifest["source_sha256"], "reused_scores": {},
                 "scope": "Early SMFA prefix only; complete experiment results remain pending"}
        runner.write_json(state_path, state)
        try:
            for module_name, metric, field in (("eval_fact", "factuality", "Factuality Score"),
                                               ("eval_meaningful", "meaningful", "Meaningful Score")):
                state["stage"] = metric
                module = importlib.import_module(module_name)
                if module.client is None:
                    raise RuntimeError("Judge credential is absent")
                module.client = module.client.with_options(timeout=120, max_retries=2)
                cache_name = f"{args.split}_generation_results_{metric}.jsonl"
                local_cache = work / cache_name
                prior = helper.verified_cache(FOLDER / cache_name, questions, field)
                if not local_cache.exists() and prior:
                    local_cache.write_text("".join(json.dumps(prior[i]) + "\n" for i in sorted(prior)))
                state["reused_scores"][metric] = len(helper.verified_cache(local_cache, questions, field))
                runner.write_json(state_path, state)
                score.score_file(module, local_input, field, metric, 4)
                state["completed_metrics"].append(metric)
                runner.write_json(state_path, state)
            audit = auditor.audit_split(work, args.split, rows)
            runner.write_json(work / "prefix_audit.json", audit)
            if helper.input_snapshot(FOLDER, args.split, args.profiles)[1] != snapshot:
                raise RuntimeError("SMFA prefix changed before publication")
            # Leave a completed late prefix isolated if the complete scorer can be writing.
            progress = generation_progress()
            published = ready(progress, args.split, args.profiles)
            additions = {}
            if published:
                for metric, field in (("factuality", "Factuality Score"), ("meaningful", "Meaningful Score")):
                    cache_name = f"{args.split}_generation_results_{metric}.jsonl"
                    additions[metric] = helper.append_verified_cache(work / cache_name, FOLDER / cache_name, questions, field)
            state.update(state="complete", stage="complete", completed_at=runner.now(),
                         published=published, appended_scores=additions,
                         audit_sha256=sha(work / "prefix_audit.json"))
            runner.write_json(state_path, state)
            print(f"Verified {len(questions)} SMFA {args.split} questions; published={published}", flush=True)
        except BaseException as error:
            state.update(state="failed", failed_at=runner.now(), error=str(error))
            runner.write_json(state_path, state)
            raise


def queue():
    with (RUN / "early_smfa_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify_sources()
        own_hash = sha(Path(__file__))
        state = {"state": "running", "pid": os.getpid(), "started_at": runner.now(),
                 "stage": "waiting_for_smfa_generation", "completed_targets": [],
                 "targets": [list(item) for item in TARGETS], "workers_per_metric": 4,
                 "launcher_sha256": own_hash, "helper_sha256": HELPER_HASH,
                 "auditor_sha256": AUDITOR_HASH, "source_sha256": manifest["source_sha256"]}
        runner.write_json(STATUS, state)
        try:
            for split, profiles in TARGETS:
                name = label(split, profiles)
                previous_path = RUN / f"{name}_status.json"
                if previous_path.exists():
                    previous = read(previous_path)
                    if previous["state"] == "complete" and previous.get("published"):
                        state["completed_targets"].append([split, profiles])
                        continue
                    if previous["state"] == "running" and live(previous["pid"]):
                        raise RuntimeError("A previous SMFA early judge is still running")
                    if previous["state"] == "failed":
                        raise RuntimeError("Inspect the failed SMFA prefix before retrying")
                while True:
                    verify_sources(own_hash)
                    progress = generation_progress()
                    if progress is None:
                        state.update(state="handed_off", stage="full_smfa_scoring", finished_at=runner.now())
                        runner.write_json(STATUS, state)
                        return
                    state.update(stage=progress["phase"], target=[split, profiles], generation=progress)
                    runner.write_json(STATUS, state)
                    if ready(progress, split, profiles):
                        break
                    time.sleep(30)
                command = [sys.executable, "-u", str(Path(__file__)), "--split", split,
                           "--profiles", str(profiles), "--expected-sha", own_hash]
                with (RUN / f"{name}.log").open("a") as log:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    state.update(stage="judging", child_pid=child.pid, command=command)
                    runner.write_json(STATUS, state)
                    code = child.wait()
                if code:
                    raise RuntimeError(f"SMFA prefix failed; inspect {name}_status.json")
                completed = read(previous_path)
                if completed["state"] != "complete" or set(completed["completed_metrics"]) != {"factuality", "meaningful"}:
                    raise RuntimeError("SMFA early child did not complete both metrics")
                state.pop("child_pid", None)
                if not completed["published"]:
                    state.update(state="handed_off", stage="full_smfa_scoring", finished_at=runner.now())
                    runner.write_json(STATUS, state)
                    return
                state["completed_targets"].append([split, profiles])
                runner.write_json(STATUS, state)
            state.update(state="complete", stage="complete", completed_at=runner.now())
            runner.write_json(STATUS, state)
        except BaseException as error:
            state.update(state="failed", failed_at=runner.now(), error=str(error))
            runner.write_json(STATUS, state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--split", choices=("forget", "retain"))
    parser.add_argument("--profiles", type=int)
    parser.add_argument("--expected-sha")
    args = parser.parse_args()
    os.umask(0o077)
    if args.launch:
        verify_sources()
        main_status = read(RUN / "status.json")
        if not live(main_status["pid"]):
            raise RuntimeError("Main pipeline is not running")
        raw = Path(f"/proc/{main_status['pid']}/environ").read_bytes()
        environment = {os.fsdecode(k): os.fsdecode(v) for k, v in
                       (item.split(b"=", 1) for item in raw.split(b"\0") if b"=" in item)}
        environment["CUDA_VISIBLE_DEVICES"] = ""
        with (RUN / "early_smfa_queue.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__))], env=environment,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(f"Launched early SMFA judge queue PID {child.pid}")
    elif args.split:
        if args.profiles is None or args.profiles <= 0 or not args.expected_sha:
            parser.error("Worker requires positive --profiles and --expected-sha")
        worker(args)
    else:
        queue()
