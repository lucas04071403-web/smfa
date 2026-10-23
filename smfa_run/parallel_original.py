"""Evaluate the Original comparison on an idle GPU while SMFA trains."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduce_smfa"))
import run as runner

RUN = ROOT / "smfa_run"
STAGING = runner.REPO / runner.EVAL_ROOT / ".parallel_original" / "Original"
DESTINATION = runner.REPO / runner.EVAL_ROOT / "Original"
STATUS = RUN / "parallel_original_status.json"


def main():
    lock = (RUN / "parallel_original.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.chdir(runner.REPO)
    manifest = json.loads((RUN / "manifest.json").read_text())
    for relative, expected in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Experiment source changed: {relative}")
    plan = []
    for name, command in manifest["commands"].items():
        if name not in ("evaluate_Original", "score_Original"):
            continue
        command = list(command)
        option = "--output_folder" if name.startswith("evaluate_") else "--input_folder"
        command[command.index(option) + 1] = str(STAGING)
        plan.append((name, command))
    status = {
        "pid": os.getpid(), "state": "running", "started_at": runner.now(),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "completed_stages": [], "commands": dict(plan),
        "staging": str(STAGING), "destination": str(DESTINATION),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": manifest["source_sha256"],
    }
    runner.write_json(STATUS, status)
    try:
        for name, command in plan:
            status.update(stage=name, stage_started_at=runner.now())
            runner.write_json(STATUS, status)
            print(f"{runner.now()} Starting parallel {name} on GPU {status['physical_gpu']}", flush=True)
            with (RUN / f"parallel_{name}.log").open("a") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                status["child_pid"] = child.pid
                runner.write_json(STATUS, status)
                code = child.wait()
            if code:
                raise RuntimeError(f"{name} exited with code {code}")
            canonical_root = runner.EVAL_ROOT
            try:
                runner.EVAL_ROOT = str(STAGING.parent)
                runner.check_stage(name)
            finally:
                runner.EVAL_ROOT = canonical_root
            status["completed_stages"].append(name)
            status.pop("child_pid", None)
            runner.write_json(STATUS, status)
            print(f"{runner.now()} Verified parallel {name}", flush=True)
        if not DESTINATION.exists():
            STAGING.rename(DESTINATION)
            runner.check_stage("evaluate_Original")
            runner.check_stage("score_Original")
            status.update(state="complete", promoted=True)
        else:
            status.update(state="complete", promoted=False)
        status.update(stage="complete", completed_at=runner.now())
        runner.write_json(STATUS, status)
        print(f"{runner.now()} Original comparison complete; promoted={status['promoted']}", flush=True)
    except BaseException as error:
        status.update(state="failed", failed_at=runner.now(), error=str(error))
        runner.write_json(STATUS, status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        pipeline = json.loads((RUN / "status.json").read_text())
        raw = Path(f"/proc/{pipeline['pid']}/environ").read_bytes()
        environment = {os.fsdecode(key): os.fsdecode(value) for key, value in
                       (item.split(b"=", 1) for item in raw.split(b"\0") if b"=" in item)}
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        with (RUN / "parallel_original.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())],
                                       env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Launched Original comparison PID {process.pid}; status: {STATUS}")
    else:
        main()
