"""Queue larger Original retain prefixes while generation remains active."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "smfa_run"
HELPER = RUN / "early_original_forget.py"
STATUS = RUN / "original_judge_queue_status.json"
TARGETS = (375, 500, 625, 750)
sys.path.insert(0, str(ROOT / "reproduce_smfa"))
import run as runner


def is_live(pid):
    try:
        os.kill(pid, 0)
        return not Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].startswith("Z ")
    except (ProcessLookupError, FileNotFoundError):
        return False


def generation_progress():
    status = json.loads((RUN / "parallel_original_status.json").read_text())
    if status["state"] == "failed":
        raise RuntimeError("Original controller failed; inspect its status before continuing")
    if status["state"] != "running" or status["stage"] != "evaluate_Original":
        return None
    folder = Path(status["staging"])
    if (folder / "retain_generation_results.json").exists():
        return None
    if not is_live(status["child_pid"]):
        raise RuntimeError("Original generation process exited without a stage update")
    try:
        progress = json.loads((folder / "progress.json").read_text())
    except json.JSONDecodeError:
        return -1
    if progress["split"] != "retain":
        return -1
    if progress["completed_profiles"] >= progress["total_profiles"] - 1:
        return None
    return progress["completed_profiles"]


def completed_prefix(prefix):
    path = RUN / f"early_original_retain_{prefix}_status.json"
    if not path.exists():
        return False
    status = json.loads(path.read_text())
    if status["state"] == "failed":
        raise RuntimeError(f"Early retain prefix {prefix} failed; inspect {path}")
    if status["state"] == "complete":
        if set(status["completed_metrics"]) != {"factuality", "meaningful"}:
            raise RuntimeError(f"Completed prefix {prefix} lacks both metrics")
        if status["profiles"] != prefix or status["questions"] != prefix * 19:
            raise RuntimeError(f"Completed prefix {prefix} has unexpected coverage")
        return True
    if not is_live(status["pid"]):
        raise RuntimeError(f"Early retain prefix {prefix} stopped before completion")
    return False


def main():
    os.umask(0o077)
    with (RUN / "original_judge_queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        helper_hash = hashlib.sha256(HELPER.read_bytes()).hexdigest()
        manifest = json.loads((RUN / "manifest.json").read_text())
        for relative, expected in manifest["source_sha256"].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Experiment source changed: {relative}")
        state = {
            "pid": os.getpid(), "state": "running", "started_at": runner.now(),
            "stage": "waiting_for_prefix_250", "targets": list(TARGETS),
            "completed_targets": [], "workers_per_metric": 4,
            "helper_sha256": helper_hash,
            "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_sha256": manifest["source_sha256"],
            "scope": "Early Original retain judging only; full experiment remains pending",
        }

        def handoff():
            state.update(state="handed_off", stage="full_original_scoring",
                         finished_at=runner.now())
            runner.write_json(STATUS, state)
            print("Original generation handed off; no further early tasks launched", flush=True)

        runner.write_json(STATUS, state)
        try:
            while not completed_prefix(250):
                if generation_progress() is None:
                    handoff()
                    return
                time.sleep(30)
            for prefix in TARGETS:
                if completed_prefix(prefix):
                    state["completed_targets"].append(prefix)
                    runner.write_json(STATUS, state)
                    continue
                while True:
                    progress = generation_progress()
                    if progress is None:
                        handoff()
                        return
                    state.update(stage="waiting_for_profiles", target_profiles=prefix,
                                 generated_profiles=progress)
                    runner.write_json(STATUS, state)
                    if progress >= prefix:
                        break
                    time.sleep(30)
                if hashlib.sha256(HELPER.read_bytes()).hexdigest() != helper_hash:
                    raise RuntimeError("Early judge helper changed after queue launch")
                command = [sys.executable, "-u", str(HELPER), "--split", "retain", "--profiles", str(prefix)]
                with (RUN / f"early_original_retain_{prefix}.log").open("a") as log:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                    state.update(stage="judging", child_pid=child.pid, command=command)
                    runner.write_json(STATUS, state)
                    print(f"{runner.now()} Judging retain prefix {prefix}, PID {child.pid}", flush=True)
                    code = child.wait()
                if code or not completed_prefix(prefix):
                    raise RuntimeError(f"Retain prefix {prefix} did not complete; exit code {code}")
                state["completed_targets"].append(prefix)
                state.pop("child_pid", None)
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
    args = parser.parse_args()
    if args.launch:
        status = json.loads((RUN / "parallel_original_status.json").read_text())
        if generation_progress() is None:
            raise RuntimeError("Original generation has already handed off")
        raw = Path(f"/proc/{status['pid']}/environ").read_bytes()
        environment = {os.fsdecode(k): os.fsdecode(v) for k, v in
                       (item.split(b"=", 1) for item in raw.split(b"\0") if b"=" in item)}
        environment["CUDA_VISIBLE_DEVICES"] = ""
        os.umask(0o077)
        with (RUN / "original_judge_queue.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())],
                                     env=environment, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Launched Original judge queue PID {child.pid}")
    else:
        main()
