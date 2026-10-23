"""Judge a disjoint, complete SMFA tail with the frozen official scoring code."""

import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "smfa_run"
REPO = ROOT / "S-MLLMUn"
sys.path.insert(0, str(ROOT / "reproduce_smfa"))
import run as runner
import score
import audit_final as auditor
import early_original_forget as helper
import early_smfa_queue as queue

START, STOP = 850, 950
OFFSET = START * 19
WORK = RUN / "smfa_tail_850_950"
STATUS = RUN / "smfa_tail_850_950_status.json"
DEST = REPO / runner.EVAL_ROOT / "SMFA"
METRICS = (("eval_fact", "factuality", "Factuality Score"),
           ("eval_meaningful", "meaningful", "Meaningful Score"))


def process(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


def frozen():
    return queue.verify_sources()


def snapshot():
    questions = []
    hashes = {}
    for index in range(START, STOP):
        path = DEST / ".profiles/retain" / f"{index:04d}" / "retain_generation_results.json"
        items = auditor.read_json(path)["Generation_Questions"]
        auditor.require(len(items) == 19, "Tail profile coverage differs")
        questions.extend(items)
        hashes[str(path.relative_to(ROOT))] = auditor.sha256(path)
    return questions, hashes


def prepare():
    import pandas as pd
    from evaluate import verify_questions

    frozen()
    questions, hashes = snapshot()
    dataset = REPO / "data/S-MLLMUn-data/eval_data/retain_5/train-00000-of-00001.parquet"
    rows = pd.read_parquet(dataset).iloc[START:STOP]
    verify_questions(questions, rows)
    expected = [(kind, row, item) for _, row in rows.iterrows()
                for kind, column in zip(auditor.KINDS, auditor.COLUMNS) for item in row[column]]
    for question, (kind, row, item) in zip(questions, expected):
        auditor.require(isinstance(question.get("generated_answer"), str)
                        and question["generated_answer"].strip(), "Empty tail answer")
        if kind != "text_memory":
            auditor.require((question.get("image_id"), question.get("image_type"))
                            == (row["ID"], item["Image"]), "Tail image identity differs")
    WORK.mkdir(exist_ok=True)
    source = WORK / "retain_generation_results.json"
    data = (json.dumps({"Generation_Questions": questions}, indent=4) + "\n").encode()
    if source.exists():
        auditor.require(source.read_bytes() == data, "Tail input changed")
    else:
        source.write_bytes(data)
    return questions, hashes, rows, dataset


def remap(path, questions, field):
    data = path.read_bytes()
    local = helper.verified_cache(path, questions, field)
    auditor.require(data.endswith(b"\n") and len(data.splitlines()) == len(questions)
                    and set(local) == set(range(len(questions))), "Tail cache coverage differs")
    auditor.require(all(row["result"]["Justification"].strip() for row in local.values()),
                    "Empty tail justification")
    return {index + OFFSET: {**local[index], "index": index + OFFSET} for index in sorted(local)}


def valid_shared(path, field):
    data = path.read_bytes() if path.exists() else b""
    auditor.require(not data or data.endswith(b"\n"), "Incomplete shared cache")
    records, profiles = {}, {}
    for line in data.splitlines():
        item = json.loads(line)
        index = item["index"]
        auditor.require(type(index) is int and 0 <= index < STOP * 19 and index not in records,
                        "Invalid or duplicated shared cache index")
        profile, within = divmod(index, 19)
        if profile not in profiles:
            source = DEST / ".profiles/retain" / f"{profile:04d}" / "retain_generation_results.json"
            profiles[profile] = auditor.read_json(source)["Generation_Questions"]
            auditor.require(len(profiles[profile]) == 19, "Shared profile coverage differs")
        question = profiles[profile][within]
        auditor.require(item["digest"] == score.digest(question), "Shared question digest differs")
        result = item["result"]
        auditor.require(type(result.get(field)) is int and 0 <= result[field] <= 10,
                        "Invalid shared score")
        for destination, origin in (("Question", "question"), ("Generated Answer", "generated_answer"),
                                    ("Ground Truth", "ground_truth"), ("Question Type", "question type"),
                                    ("Image_ID", "image_id"), ("image_type", "image_type")):
            auditor.require(result.get(destination) == question.get(origin), "Shared score identity differs")
        auditor.require(isinstance(result.get("Justification"), str) and result["Justification"].strip(),
                        "Empty shared justification")
        records[index] = item
    return data, records


def generation_available():
    main = auditor.read_json(RUN / "status.json")
    if main["state"] != "running" or main["stage"] != "evaluate_SMFA":
        return False
    auditor.require(queue.live(main["pid"]) and queue.live(main["child_pid"]), "Generation stopped")
    progress = auditor.read_json(DEST / "progress.json")
    return progress["split"] == "retain" and progress["completed_profiles"] < 940


def publish(state, questions):
    # The queue owns its lock for its lifetime; pause only its scheduler, not its worker.
    while generation_available():
        frozen()
        current = auditor.read_json(queue.STATUS)
        child = current.get("child_pid")
        identity = process(current["pid"])
        if current["state"] != "running" or current["stage"] != "judging" or not identity or not child:
            time.sleep(10)
            continue
        split, profiles = current["target"]
        worker_state = auditor.read_json(RUN / f"early_smfa_{split}_{profiles}_status.json")
        cache = RUN / f"early_smfa_{split}_{profiles}/{split}_generation_results_meaningful.jsonl"
        if worker_state.get("stage") != "meaningful" or not cache.exists():
            time.sleep(10)
            continue
        remaining = profiles * 19 - len(cache.read_bytes().splitlines())
        if remaining > 20:
            time.sleep(10)
            continue
        scheduler = current["pid"]
        guard = subprocess.Popen([sys.executable, str(Path(__file__)), "--resume-guard",
                                  str(scheduler), identity[1], str(os.getpid())],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
        stopped = False
        try:
            auditor.require(process(scheduler) and process(scheduler)[1] == identity[1],
                            "Queue identity changed")
            os.kill(scheduler, signal.SIGSTOP)
            stopped = True
            deadline = time.monotonic() + 180
            while process(scheduler) and process(scheduler)[0] != "T" and time.monotonic() < deadline:
                time.sleep(0.05)
            auditor.require(process(scheduler) == ("T", identity[1]), "Queue did not pause")
            latest = auditor.read_json(queue.STATUS)
            auditor.require(latest.get("child_pid") == child, "Queue child changed before pause")
            state.update(stage="waiting_for_current_prefix_to_close", scheduler_pid=scheduler,
                         scheduler_start_ticks=identity[1], worker_pid=child, checked_at=auditor.now())
            auditor.write_json(STATUS, state)
            while queue.live(child) and time.monotonic() < deadline:
                time.sleep(1)
            if queue.live(child):
                continue
            worker_state = auditor.read_json(RUN / f"early_smfa_{split}_{profiles}_status.json")
            auditor.require(worker_state["state"] == "complete" and worker_state.get("published"),
                            "Current prefix did not publish successfully")
            with (RUN / "early_smfa.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                auditor.require(generation_available(), "Full scoring handoff reached; retain isolated tail")
                auditor.require(snapshot()[0] == questions, "Tail generation changed")
                publications = {}
                for _, metric, field in METRICS:
                    shared = DEST / f"retain_generation_results_{metric}.jsonl"
                    before, existing = valid_shared(shared, field)
                    mapped = remap(WORK / shared.name, questions, field)
                    auditor.require(all(existing[index] == mapped[index] for index in existing.keys() & mapped.keys()),
                                    "Conflicting shared tail score")
                    additions = [mapped[i] for i in sorted(mapped.keys() - existing.keys())]
                    expected = before + b"".join((json.dumps(item) + "\n").encode() for item in additions)
                    auditor.require(shared.read_bytes() == before, "Shared cache changed during closed publication")
                    temporary = shared.with_suffix(".tail.tmp")
                    with temporary.open("xb") as handle:
                        handle.write(expected)
                        handle.flush()
                        os.fsync(handle.fileno())
                    temporary.replace(shared)
                    after, actual = valid_shared(shared, field)
                    auditor.require(after == expected and all(actual.get(i) == item for i, item in mapped.items()),
                                    "Tail publication differs")
                    publications[metric] = {"added_records": len(additions), "previous_records": len(existing),
                                            "shared_sha256": auditor.sha256(shared),
                                            "previous_record_bytes_preserved": after.startswith(before)}
                state.update(state="complete", stage="complete", published=True,
                             publications=publications, completed_at=auditor.now())
                auditor.write_json(STATUS, state)
                return
        finally:
            if stopped and process(scheduler) and process(scheduler)[1] == identity[1]:
                os.kill(scheduler, signal.SIGCONT)
            guard.terminate()
            guard.wait()
    state.update(state="complete", stage="isolated_after_handoff", published=False, completed_at=auditor.now())
    auditor.write_json(STATUS, state)


def main(prepare_only=False):
    os.umask(0o077)
    with (RUN / "smfa_tail_850_950.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        auditor.require(not STATUS.exists(), "Inspect the existing tail scorer before resuming")
        questions, hashes, rows, dataset = prepare()
        auditor.require(len(questions) == 1900, "Tail question coverage differs")
        if prepare_only:
            print(json.dumps({"state": "prepared", "questions": len(questions), "api_calls": 0,
                              "input_sha256": auditor.sha256(WORK / "retain_generation_results.json")}))
            return
        auditor.require(generation_available(), "Generation is not available for early tail scoring")
        for _, metric, field in METRICS:
            _, existing = valid_shared(DEST / f"retain_generation_results_{metric}.jsonl", field)
            auditor.require(not any(index >= OFFSET for index in existing), "Tail scores already exist")
        state = {"state": "running", "stage": "preparing", "pid": os.getpid(),
                 "started_at": auditor.now(), "start_profile": START, "stop_profile": STOP,
                 "questions": len(questions), "workers_per_metric": 4, "completed_metrics": [],
                 "profile_sha256": hashes, "data_sha256": auditor.sha256(dataset),
                 "input_sha256": auditor.sha256(WORK / "retain_generation_results.json"),
                 "source_sha256": frozen()["source_sha256"],
                 "entrypoint_sha256": auditor.sha256(Path(__file__)), "published": False}
        auditor.write_json(STATUS, state)
        try:
            for name, metric, field in METRICS:
                frozen()
                module = importlib.import_module(name)
                auditor.require(module.client is not None, "Judge credential is absent")
                module.client = module.client.with_options(timeout=120, max_retries=2)
                state.update(stage=metric)
                auditor.write_json(STATUS, state)
                score.score_file(module, WORK / "retain_generation_results.json", field, metric, 4)
                state["completed_metrics"].append(metric)
                auditor.write_json(STATUS, state)
            measured = auditor.audit_split(WORK, "retain", rows)
            auditor.write_json(WORK / "tail_audit.json", measured)
            state.update(stage="waiting_for_publication", audit_sha256=auditor.sha256(WORK / "tail_audit.json"))
            auditor.write_json(STATUS, state)
            publish(state, questions)
        except BaseException as error:
            state.update(state="failed", error=str(error), failed_at=auditor.now())
            auditor.write_json(STATUS, state)
            raise


def resume_guard(arguments):
    scheduler, start, parent = int(arguments[0]), arguments[1], int(arguments[2])
    deadline = time.monotonic() + 240
    while process(parent) and process(parent)[0] != "Z" and time.monotonic() < deadline:
        time.sleep(5)
    current = process(scheduler)
    if current and current[1] == start:
        os.kill(scheduler, signal.SIGCONT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume-guard", nargs=3)
    args = parser.parse_args()
    if args.resume_guard:
        resume_guard(args.resume_guard)
    elif args.launch:
        manifest = frozen()
        pipeline = auditor.read_json(RUN / "status.json")
        auditor.require(queue.live(pipeline["pid"]), "Formal pipeline controller is absent")
        raw = Path(f"/proc/{pipeline['pid']}/environ").read_bytes()
        environment = {os.fsdecode(k): os.fsdecode(v) for k, v in
                       (item.split(b"=", 1) for item in raw.split(b"\0") if b"=" in item)}
        environment["CUDA_VISIBLE_DEVICES"] = ""
        os.umask(0o077)
        with (RUN / "smfa_tail_850_950.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())], env=environment,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(f"Launched disjoint SMFA tail scorer PID {child.pid}")
    else:
        main(args.prepare_only)
