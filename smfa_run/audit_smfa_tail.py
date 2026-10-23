"""Independently verify completed tail scores and their shared-cache publication."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import audit_final as auditor
import audit_smfa_prefix as prefix

RUN = auditor.RUN
DEST = auditor.EVALUATION / "SMFA"
WORK = RUN / "smfa_tail_850_950"
STATUS = RUN / "smfa_tail_850_950_status.json"
OUTPUT = RUN / "verification_smfa_tail_850_950.json"
WATCH_STATUS = RUN / "smfa_tail_audit_status.json"
OFFSET, COUNT = 850 * 19, 100 * 19


def process(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


def verify_publication(local_path, shared_path, previous_path, metadata):
    local = prefix.records(local_path)
    previous = prefix.records(previous_path)
    shared = prefix.records(shared_path)
    auditor.require(set(local) == set(range(COUNT)), "Tail local coverage differs")
    auditor.require(type(metadata["previous_records"]) is int
                    and set(previous) == set(range(metadata["previous_records"])),
                    "Previous published prefix coverage differs")
    auditor.require(metadata["added_records"] == COUNT
                    and metadata["previous_record_bytes_preserved"] is True,
                    "Tail publication metadata differs")
    auditor.require(len(previous) <= 750 * 19, "Tail overlaps the previous prefix")
    auditor.require(all(0 <= index < 950 * 19 for index in shared),
                    "Shared score index is outside the retain split")
    before = b"".join(previous[index] for index in sorted(previous))
    additions = []
    for index in range(COUNT):
        expected = {**json.loads(local[index]), "index": index + OFFSET}
        auditor.require(index + OFFSET in shared
                        and json.loads(shared[index + OFFSET]) == expected,
                        f"Mapped tail record differs: {index}")
        additions.append((json.dumps(expected) + "\n").encode())
    published = before + b"".join(additions)
    current = b"".join(shared.values())
    auditor.require(current.startswith(published), "Previous cache or tail publication bytes differ")
    auditor.require(hashlib.sha256(published).hexdigest() == metadata["shared_sha256"],
                    "Publication snapshot hash differs")
    return {"mapped_records_verified": COUNT, "previous_record_bytes_verified": len(previous),
            "publication_sha256": hashlib.sha256(published).hexdigest(),
            "previous_record_bytes_sha256": hashlib.sha256(before).hexdigest(),
            "shared_snapshot_records": len(shared),
            "shared_snapshot_sha256": hashlib.sha256(current).hexdigest()}


def audit():
    import pandas as pd

    manifest = auditor.read_json(RUN / "manifest.json")
    auditor.require(sys.prefix == manifest["conda_prefix"], "Use the recorded Conda environment")
    for name, expected in manifest["source_sha256"].items():
        auditor.require(auditor.sha256(auditor.ROOT / name) == expected, f"Frozen source changed: {name}")
    state = auditor.read_json(STATUS)
    auditor.require(state["state"] == "complete"
                    and set(state["completed_metrics"]) == {"factuality", "meaningful"},
                    "Tail scoring is incomplete")
    auditor.require((state["start_profile"], state["stop_profile"], state["questions"])
                    == (850, 950, COUNT), "Tail scope differs")
    auditor.require(state["source_sha256"] == manifest["source_sha256"], "Tail source manifest differs")
    auditor.require(auditor.sha256(RUN / "score_smfa_tail.py") == state["entrypoint_sha256"],
                    "Tail scorer source changed")
    auditor.require(auditor.sha256(WORK / "retain_generation_results.json") == state["input_sha256"],
                    "Tail judged input changed")
    auditor.require(auditor.sha256(WORK / "tail_audit.json") == state["audit_sha256"],
                    "Tail worker audit changed")
    dataset = auditor.REPO / "data/S-MLLMUn-data/eval_data/retain_5/train-00000-of-00001.parquet"
    auditor.require(auditor.sha256(dataset) == state["data_sha256"], "Tail dataset changed")
    rows = pd.read_parquet(dataset)
    measured = auditor.audit_split(WORK, "retain", rows.iloc[850:950])
    source_questions, source_hashes = [], {}
    for index in range(850, 950):
        path = DEST / ".profiles/retain" / f"{index:04d}" / "retain_generation_results.json"
        name = str(path.relative_to(auditor.ROOT))
        source_hashes[name] = auditor.sha256(path)
        auditor.require(source_hashes[name] == state["profile_sha256"][name], "Tail profile changed")
        questions = auditor.read_json(path)["Generation_Questions"]
        auditor.require(len(questions) == 19, "Tail source profile coverage differs")
        source_questions.extend(questions)
    auditor.require(set(source_hashes) == set(state["profile_sha256"]), "Tail profile manifest differs")
    auditor.require(source_questions == auditor.read_json(WORK / "retain_generation_results.json")["Generation_Questions"],
                    "Tail source profiles differ from judged answers")
    result = {"state": "passed", "checked_at": auditor.now(),
              "scope": "Disjoint SMFA retain tail only; full experiment remains pending",
              "generated_answers_verified": COUNT, "judge_scores_verified": COUNT * 2,
              "split": measured, "profile_sha256": source_hashes,
              "published": state["published"], "publications": {},
              "source_sha256": manifest["source_sha256"],
              "tail_scorer_sha256": state["entrypoint_sha256"],
              "entrypoint_sha256": auditor.sha256(Path(__file__)),
              "api_calls": 0, "gpu_allocations": 0, "formal_inputs_modified": False}
    if state["published"]:
        for metric in ("factuality", "meaningful"):
            metadata = state["publications"][metric]
            count = metadata["previous_records"]
            auditor.require(type(count) is int and count % 19 == 0, "Previous prefix size differs")
            profiles = count // 19
            report_path = RUN / f"verification_smfa_retain_{profiles}.json"
            if not report_path.exists():
                report = prefix.audit(profiles)
                auditor.write_json(report_path, report)
            report = auditor.read_json(report_path)
            auditor.require(report["state"] == "passed" and report["profiles"] == profiles,
                            "Previous prefix independent audit is unavailable")
            previous_folder = RUN / f"early_smfa_retain_{profiles}"
            for name, expected in report["split"]["sha256"].items():
                auditor.require(auditor.sha256(previous_folder / name) == expected,
                                "Previous independently audited prefix changed")
            name = f"retain_generation_results_{metric}.jsonl"
            result["publications"][metric] = verify_publication(
                WORK / name, DEST / name, previous_folder / name, metadata)
        queue = auditor.read_json(RUN / "early_smfa_queue_status.json")
        identity = process(state["scheduler_pid"])
        auditor.require(not identity or identity[1] != state["scheduler_start_ticks"]
                        or identity[0] not in ("T", "t"), "Tail publication left its scheduler paused")
        auditor.require(queue["state"] in ("running", "complete", "handed_off"), "Queue failed after publication")
        if queue["state"] == "running":
            current = process(queue["pid"])
            auditor.require(current and current[0] not in ("Z", "T", "t"), "Active scheduler is unavailable")
        result["scheduler_resumed_verified"] = True
        result["scheduler_snapshot"] = {"state": queue["state"], "stage": queue["stage"], "pid": queue["pid"]}
    for name, expected in measured["sha256"].items():
        auditor.require(auditor.sha256(WORK / name) == expected, "Tail artifact changed during audit")
    result["tail_status_sha256"] = auditor.sha256(STATUS)
    output = OUTPUT if state["published"] else OUTPUT.with_name("verification_smfa_tail_850_950_isolated.json")
    auditor.require(not output.exists(), "Refusing to replace an existing independent audit")
    auditor.write_json(output, result)
    print(json.dumps({"output": str(output), "state": result["state"],
                      "judge_scores_verified": result["judge_scores_verified"], "published": result["published"]}), flush=True)


def watch():
    os.umask(0o077)
    with (RUN / "smfa_tail_audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"state": "running", "pid": os.getpid(), "started_at": auditor.now(),
                 "stage": "waiting_for_tail", "entrypoint_sha256": auditor.sha256(Path(__file__))}
        auditor.write_json(WATCH_STATUS, state)
        try:
            while True:
                tail = auditor.read_json(STATUS)
                if tail["state"] == "complete":
                    break
                worker = process(tail["pid"])
                auditor.require(tail["state"] == "running" and worker and worker[0] != "Z",
                                "Tail scorer failed or stopped; inspect its preserved caches")
                time.sleep(30)
            audit()
            state.update(state="complete", stage="complete", completed_at=auditor.now())
        except BaseException as error:
            state.update(state="failed", error=str(error), failed_at=auditor.now())
            raise
        finally:
            auditor.write_json(WATCH_STATUS, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        auditor.require(not WATCH_STATUS.exists(), "Inspect the existing tail audit watcher")
        worker = auditor.read_json(STATUS)
        identity = process(worker["pid"])
        auditor.require(worker["state"] == "running" and identity and identity[0] != "Z", "Tail scorer is absent")
        os.umask(0o077)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        with (RUN / "smfa_tail_audit.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--watch"],
                                     env=environment, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Launched independent tail audit watcher PID {child.pid}")
    elif args.watch:
        watch()
    else:
        audit()
