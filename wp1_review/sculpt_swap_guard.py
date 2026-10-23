"""Provide temporary swap for one official sculpt process, then remove it."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

RUN = Path(__file__).resolve().parent / "step0_baselines"


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--gib", type=int, default=64)
    args = parser.parse_args()
    identity = process_identity(args.pid)
    if identity is None:
        raise SystemExit("The targeted sculpt process has already exited")
    swap = RUN / f"official_sculpt_{args.pid}.swap"
    record_path = RUN / f"official_sculpt_swap_{args.pid}.json"
    record = {"state": "preparing", "pid": args.pid, "start_ticks": identity,
              "size_gib": args.gib, "swap_file": str(swap),
              "started_at": datetime.now(timezone.utc).isoformat(),
              "reason": "Official sculptor retains three dense updates, masks and two pruned copies; 128 GiB RAM alone is insufficient",
              "cleanup": "swapoff and unlink this temporary file after the targeted process exits"}

    def save():
        record_path.write_text(json.dumps(record, indent=2) + "\n")

    save()
    active = False
    try:
        descriptor = os.open(swap, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        subprocess.run(["fallocate", "-l", f"{args.gib}G", str(swap)], check=True)
        subprocess.run(["mkswap", str(swap)], check=True)
        subprocess.run(["swapon", str(swap)], check=True)
        active = True
        record.update(state="active", activated_at=datetime.now(timezone.utc).isoformat())
        save()
        if process_identity(args.pid) == identity:
            os.kill(args.pid, signal.SIGCONT)
        while process_identity(args.pid) == identity:
            time.sleep(5)
        record["sculpt_process_exited_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as error:
        record.update(state="failed", error=str(error))
        # Do not let the paused, unprotected sculptor exhaust machine memory.
        if not active and process_identity(args.pid) == identity:
            os.kill(args.pid, signal.SIGTERM)
            os.kill(args.pid, signal.SIGCONT)
        raise
    finally:
        if active:
            result = subprocess.run(["swapoff", str(swap)], capture_output=True, text=True)
            record["swapoff_exit_code"] = result.returncode
            if result.returncode:
                record.update(state="cleanup_failed", cleanup_error=result.stderr[-2000:])
                save()
                raise RuntimeError("Temporary swap remains active; see " + str(record_path))
        if swap.exists():
            swap.unlink()
        if record["state"] != "failed":
            record["state"] = "complete"
        record["ended_at"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
