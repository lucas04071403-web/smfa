"""Resume failed judge calls without interrupting the active baseline runner.

The original scorer, judge prompts and cached valid scores remain unchanged.
Recovery writes a sidecar until the original runner releases its process lock.
"""

import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import run_step0_baselines as runner


RECOVERY = runner.RUN / "scoring_recovery.json"
FIELDS = {"factuality": "Factuality Score", "meaningful": "Meaningful Score"}


def persist(data):
    temporary = RECOVERY.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(RECOVERY)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sources(status):
    changes = {}
    for name, expected in status["source_sha256"].items():
        actual = fingerprint(runner.ROOT / name)
        if actual == expected:
            continue
        if name != "wp1_review/memory_construct.py":
            raise RuntimeError(f"Pinned experiment source changed: {name}")
        proof = json.loads((runner.ROOT / "wp1_review/memory_construct_precision_correction.json").read_text())
        if (proof["before_sha256"], proof["after_sha256"], proof["state"]) != (expected, actual, "passed"):
            raise RuntimeError("Memory self-check correction provenance does not match")
        changes[name] = {"before": expected, "after": actual,
                         "note": "Documented CPU self-check audit correction; unused by scoring recovery"}
    return changes


def audit_scores(label):
    folder = runner.REPO / runner.EVAL_ROOT / label
    summary = {}
    for split, expected_count in (("forget", 950), ("retain", 18050)):
        questions = json.loads((folder / f"{split}_generation_results.json").read_text())["Generation_Questions"]
        if len(questions) != expected_count:
            raise RuntimeError(f"{label}/{split}: unexpected generation count")
        for suffix, field in FIELDS.items():
            cache = folder / f"{split}_generation_results_{suffix}.jsonl"
            records = {}
            for line in cache.read_text().splitlines():
                row = json.loads(line)
                index = row["index"]
                if not 0 <= index < len(questions):
                    raise RuntimeError("Judge cache index is out of bounds")
                digest = hashlib.sha256(json.dumps(questions[index], sort_keys=True).encode()).hexdigest()
                if row["digest"] != digest:
                    raise RuntimeError("Judge cache refers to different generated content")
                records[index] = row["result"]
            valid = sum(type(r.get(field)) is int and 0 <= r[field] <= 10 for r in records.values())
            if len(records) != expected_count or valid != expected_count:
                raise RuntimeError(f"{label}/{split}/{suffix}: only {valid}/{expected_count} valid scores")
            output = json.loads((folder / f"{split}_generation_results_{suffix}_score.json").read_text())
            for kind in ("image_understanding", "image_memory", "text_memory"):
                expected = [records[i] for i, q in enumerate(questions) if q["question type"] == kind]
                if output[kind] != expected:
                    raise RuntimeError(f"{label}/{split}/{suffix}: exported scores differ from cache")
            summary[f"{split}_{suffix}"] = {"valid": valid, "expected": expected_count}
    return summary


def score_with_retries(data, label, original):
    command = [str(runner.PYTHON), "-u", str(runner.ROOT / "reproduce_smfa/score.py"),
               "--input_folder", f"{runner.EVAL_ROOT}/{label}", "--workers", "4"]
    past = sum(s["stage"] == "score" for s in original[label]["stages"])
    model = data["models"].setdefault(label, {"state": "running", "stages": []})
    for retry in range(1, 6):
        check_sources(original)
        number = past + retry
        log_path = runner.RUN / f"{label}_score_{number}.log"
        record = {"stage": "score", "state": "running", "gpu": 0 if label == "Base" else 1,
                  "started_at": runner.now(), "attempt": number, "command": command,
                  "log": str(log_path), "recovery": "cached missing scores only"}
        model["stages"].append(record)
        data["state"] = f"scoring_{label}"
        persist(data)
        print(f"Resume {label}, attempt {number}: {log_path}", flush=True)
        start = time.monotonic()
        with log_path.open("x") as log:
            child = subprocess.Popen(command, cwd=runner.REPO,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                                     stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = child.pid
            persist(data)
            exit_code = child.wait()
        record.update(ended_at=runner.now(), duration_seconds=round(time.monotonic()-start, 3),
                      exit_code=exit_code, state="complete" if exit_code == 0 else "failed")
        persist(data)
        if exit_code == 0:
            model["audit"] = audit_scores(label)
            model["state"] = "complete"
            persist(data)
            return
        print(f"{label} attempt {number} failed; retrying missing cached scores", flush=True)
        if retry < 5:
            time.sleep(30)
    model["state"] = "failed"
    persist(data)


def main():
    lock = (runner.RUN / "scoring_recovery.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if RECOVERY.exists():
        raise RuntimeError("Recovery evidence already exists; inspect before another launch")
    initial = json.loads((runner.RUN / "status.json").read_text())
    if initial["MFA_unsculpted"]["state"] != "failed":
        raise RuntimeError("MFA must be failed, with no active scorer, to start this recovery")
    for label in ("Base", "MFA_unsculpted"):
        if not any(s["stage"] == "evaluate" and s["state"] == "complete" for s in initial[label]["stages"]):
            raise RuntimeError("Both evaluations must be complete")
    data = {"state": "starting", "started_at": runner.now(), "pid": os.getpid(), "models": {},
            "previous_runner_pid": initial["runner_pid"], "prior_errors": initial.get("errors", []),
            "source_delta_audit": check_sources(initial), "script_sha256": fingerprint(Path(__file__)),
            "timing_policy": "Sum actual subprocess attempts; exclude downtime between retries"}
    (runner.RUN / "status_before_scoring_recovery.json").write_text(json.dumps(initial, indent=2)+"\n")
    persist(data)
    try:
        data["network"] = runner.configure_network_environment()
        data["credential_source"] = runner.load_scoring_credentials()
        score_with_retries(data, "MFA_unsculpted", initial)
        data["state"] = "waiting_for_original_runner"
        persist(data)
        # Never overwrite the original runner's in-memory status or touch its scorer.
        original_lock = (runner.RUN / "runner.lock").open("a+")
        while True:
            try:
                fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                data["checked_at"] = runner.now()
                persist(data)
                time.sleep(30)
        status = json.loads((runner.RUN / "status.json").read_text())
        (runner.RUN / "status_original_runner_finished.json").write_text(json.dumps(status, indent=2)+"\n")
        check_sources(status)
        if status["Base"]["state"] == "failed":
            score_with_retries(data, "Base", status)
        elif status["Base"]["state"] == "complete":
            data["base_audit"] = audit_scores("Base")
        else:
            raise RuntimeError("Original runner ended without a terminal Base state; inspect before resuming")
        for label, model in data["models"].items():
            status[label]["stages"].extend(model["stages"])
            status[label]["state"] = model["state"]
        previous_errors = status.get("errors", [])
        resolved = [e for e in previous_errors if e.get("model") in data["models"]
                    and data["models"][e["model"]]["state"] == "complete"
                    and "/score failed" in e.get("error", "")]
        status.setdefault("historical_errors", []).extend(resolved)
        status["errors"] = [e for e in previous_errors if e not in resolved]
        status.update(state="postprocessing", runner_pid=os.getpid(), scoring_recovery=str(RECOVERY))
        runner.save_status(status)
        start = time.monotonic()
        result = subprocess.run([str(runner.PYTHON), str(runner.ROOT / "wp1_review/prepare_wp1.py"),
                                 "--analyze-existing"], cwd=runner.ROOT)
        status["postprocess_exit_code"] = result.returncode
        status["postprocess_duration_seconds"] = round(time.monotonic()-start, 3)
        if result.returncode:
            status["errors"].append({"stage": "postprocess", "error": "Split analysis failed after scoring recovery"})
        complete = all(status[label]["state"] == "complete" for label in ("Base", "MFA_unsculpted"))
        status.update(state="complete" if complete and not status["errors"] else "failed", ended_at=runner.now())
        runner.save_status(status)
        runner.write_timings(status)
        data.update(state=status["state"], ended_at=runner.now())
        persist(data)
        print(f"Recovery {data['state']}", flush=True)
    except BaseException as error:
        data.update(state="failed", error=str(error), ended_at=runner.now())
        persist(data)
        raise


if __name__ == "__main__":
    main()
