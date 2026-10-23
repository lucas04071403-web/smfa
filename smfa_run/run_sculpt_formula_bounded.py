"""Run the unchanged formula audit with bounded spare capacity on GPU 0."""

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import audit_final as auditor
import audit_sculpt_formula as formula


def process_identity(pid):
    folder = Path(f"/proc/{pid}")
    fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
    return fields[19], (folder / "cmdline").read_bytes().split(b"\0")


def main():
    os.umask(0o077)
    run = auditor.RUN
    manifest = auditor.read_json(run / "manifest.json")
    auditor.require(sys.prefix == manifest["conda_prefix"], "Activate the recorded Conda environment")
    auditor.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Use physical GPU 0")
    for relative, expected in manifest["source_sha256"].items():
        auditor.require(auditor.sha256(auditor.ROOT / relative) == expected,
                        f"Experiment source changed: {relative}")
    status = run / "sculpt_formula_audit_status.json"
    previous = auditor.read_json(status)
    auditor.require(previous["state"] == "running" and previous["stage"] == "waiting_for_idle_gpu_0",
                    "The existing formula auditor is not idle")
    source = Path(formula.__file__).resolve()
    auditor.require(auditor.sha256(source) == previous["auditor_sha256"], "Formula audit source changed")
    auditor.require(not (run / "sculpt_formula_audit.json").exists(), "Formula audit already exists")
    watcher_pid = previous["pid"]
    identity = process_identity(watcher_pid)
    auditor.require(str(source).encode() in identity[1] and b"--watch" in identity[1],
                    "Unexpected idle watcher process")
    for sample in range(3):
        rows = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits",
        ], text=True).splitlines()
        row = next(line for line in rows if line.split(",")[0].strip() == "0")
        _, free, utilization = (int(part.strip()) for part in row.split(","))
        auditor.require(free >= 20 * 1024 and utilization <= 10,
                        "GPU 0 does not have measured spare capacity")
        print(f"GPU 0 preflight {sample + 1}: {free} MiB free, {utilization}% utilization", flush=True)
        if sample < 2:
            time.sleep(5)
    stamp = auditor.now().replace(":", "").replace("+", "_")
    evidence = run / f"sculpt_formula_bounded_launch_{stamp}"
    evidence.mkdir()
    for name in ("sculpt_formula_audit_status.json", "sculpt_formula_audit.log"):
        path = run / name
        if path.exists():
            shutil.copy2(path, evidence / name)
    latest = auditor.read_json(status)
    auditor.require(latest["pid"] == watcher_pid and latest["stage"] == "waiting_for_idle_gpu_0",
                    "The watcher began work during preparation")
    auditor.require(process_identity(watcher_pid) == identity, "The watcher process identity changed")
    record = {
        "state": "starting", "checked_at": auditor.now(), "pid": os.getpid(),
        "prior_waiting_watcher_pid": watcher_pid, "physical_gpu": 0,
        "observed_free_MiB": free, "torch_allocator_limit_bytes": 4 * 1024 ** 3,
        "purpose": "Run the unchanged full formula audit using measured spare GPU capacity",
        "auditor_sha256": previous["auditor_sha256"],
        "launcher_sha256": auditor.sha256(Path(__file__)),
        "source_sha256": manifest["source_sha256"], "formal_experiment_processes_untouched": True,
    }
    auditor.write_json(evidence / "launch.json", record)
    try:
        os.kill(watcher_pid, signal.SIGTERM)
        for _ in range(100):
            if not auditor.live(watcher_pid):
                break
            time.sleep(0.1)
        auditor.require(not auditor.live(watcher_pid), "The idle watcher did not exit")
        print(f"Replaced idle formula watcher {watcher_pid}; starting the unchanged complete audit", flush=True)
        import torch

        torch.set_num_threads(4)
        limit = record["torch_allocator_limit_bytes"]
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(limit / total, device=0)
        record.update(state="auditing", auditing_started_at=auditor.now())
        auditor.write_json(evidence / "launch.json", record)
        formula.main(False)
        result_path = run / "sculpt_formula_audit.json"
        result = auditor.read_json(result_path)
        auditor.require(result["state"] == "passed" and result["exactly_reconstructed_layers"] == 274
                        and result["all_adapted_weights_exact_equal"], "The full formula audit did not pass")
        record.update(state="complete", completed_at=auditor.now(),
                      max_torch_memory_allocated_bytes=torch.cuda.max_memory_allocated(0),
                      max_torch_memory_reserved_bytes=torch.cuda.max_memory_reserved(0),
                      formula_audit_sha256=auditor.sha256(result_path),
                      canonical_status_sha256=auditor.sha256(status), api_calls=0)
        auditor.write_json(evidence / "launch.json", record)
        print(f"All 274 layers reconstructed exactly; evidence: {result_path}", flush=True)
        print(f"Peak PyTorch allocation: {record['max_torch_memory_allocated_bytes']} bytes", flush=True)
    except BaseException as error:
        record.update(state="failed", failed_at=auditor.now(), error=str(error))
        auditor.write_json(evidence / "launch.json", record)
        raise


if __name__ == "__main__":
    main()
