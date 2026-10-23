"""Finish the two MFA judge records rejected by the official JSON parser."""

import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import recover_step0_scoring as recovery


RUN = recovery.runner.RUN
STATE = RUN / "final_score_recovery.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def save(state):
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATE)


def main():
    lock = (RUN / "final_score_recovery.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = json.loads((RUN / "scoring_recovery.json").read_text())
    if old["state"] != "failed" or old.get("models", {}).get("MFA_unsculpted", {}).get("state") != "failed":
        raise RuntimeError("Expected the exhausted, failed MFA recovery state")
    original = json.loads((RUN / "status.json").read_text())
    if original.get("Base", {}).get("state") != "complete":
        raise RuntimeError("Base must remain complete before recovering MFA")
    recovery.check_sources(original)
    state = {"state": "running", "started_at": now(), "pid": os.getpid(),
             "repair": "remove only invalid JSON escaping of apostrophes from score explanations",
             "scoring_model_prompt_and_rubric": "unchanged official qwen-plus scorer",
             "prior_attempts": old["models"]["MFA_unsculpted"]["stages"], "stages": []}
    save(state)
    recovery.runner.configure_network_environment()
    state["credential_source"] = recovery.runner.load_scoring_credentials()
    command = [str(recovery.runner.PYTHON), "-u",
               str(recovery.runner.ROOT / "wp1_review/score_json_recovery_runner.py"),
               "--input_folder", f"{recovery.runner.EVAL_ROOT}/MFA_unsculpted", "--workers", "4"]
    for attempt in range(1, 6):
        record = {"attempt": attempt, "state": "running", "started_at": now(),
                  "command": command, "log": str(RUN / f"MFA_unsculpted_score_json_recovery_{attempt}.log")}
        state["stages"].append(record)
        save(state)
        start = time.monotonic()
        with Path(record["log"]).open("x") as log:
            result = subprocess.run(command, cwd=recovery.runner.REPO,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                                    stdout=log, stderr=subprocess.STDOUT)
        record.update(ended_at=now(), duration_seconds=round(time.monotonic()-start, 3),
                      exit_code=result.returncode,
                      state="complete" if result.returncode == 0 else "failed")
        save(state)
        if result.returncode == 0:
            state["audit"] = recovery.audit_scores("MFA_unsculpted")
            break
        time.sleep(30)
    else:
        state.update(state="failed", ended_at=now(), error="Official score validation still has missing scores")
        save(state)
        return

    # Wait until the original runner has written its terminal status before merging.
    original_lock = (RUN / "runner.lock").open("a+")
    while True:
        try:
            fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(10)
    status = json.loads((RUN / "status.json").read_text())
    recovery.check_sources(status)
    state_before = status["state"]
    previous_errors = status.get("errors", [])
    status.setdefault("historical_errors", []).extend(previous_errors)
    status["errors"] = []
    existing_stages = status.setdefault("MFA_unsculpted", {}).setdefault("stages", [])
    existing_keys = {(s.get("stage"), s.get("log"), s.get("started_at")) for s in existing_stages}
    for prior in old["models"]["MFA_unsculpted"]["stages"]:
        key = (prior.get("stage"), prior.get("log"), prior.get("started_at"))
        if key not in existing_keys:
            existing_stages.append(prior)
            existing_keys.add(key)
    for stage in state["stages"]:
        status["MFA_unsculpted"]["stages"].append({
            "stage": "score", "gpu": 1, "started_at": stage["started_at"],
            "ended_at": stage.get("ended_at"), "duration_seconds": stage.get("duration_seconds"),
            "state": stage["state"], "attempt": 1 + sum(s.get("stage") == "score" for s in status["MFA_unsculpted"]["stages"]),
            "command": stage["command"],
            "log": stage["log"], "exit_code": stage["exit_code"],
            "json_apostrophe_escape_repair": True})
    status["MFA_unsculpted"]["state"] = "complete"
    status["state"] = "postprocessing"
    status["scoring_recovery"] = str(STATE)
    recovery.runner.save_status(status)
    post_start = time.monotonic()
    post = subprocess.run([str(recovery.runner.PYTHON),
                           str(recovery.runner.ROOT / "wp1_review/prepare_wp1.py"),
                           "--analyze-existing"], cwd=recovery.runner.ROOT)
    status["postprocess_exit_code"] = post.returncode
    status["postprocess_duration_seconds"] = round(time.monotonic()-post_start, 3)
    if post.returncode:
        status["errors"] = [{"stage": "postprocess", "error": "Split analysis failed"}]
        status["state"] = "failed"
    else:
        status["state"] = "complete"
        status["ended_at"] = now()
    recovery.runner.save_status(status)
    recovery.runner.write_timings(status)
    state.update(state=status["state"], ended_at=now(),
                 original_runner_terminal_state=state_before,
                 postprocess_exit_code=post.returncode)
    save(state)


if __name__ == "__main__":
    main()
