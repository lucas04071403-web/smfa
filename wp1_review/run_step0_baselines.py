"""Run Base and unsculpted MFA step-0 pipelines on two GPUs with timings."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "S-MLLMUn"
RUN = ROOT / "wp1_review/step0_baselines"
PYTHON = Path("/root/anaconda3/envs/smllmun/bin/python")
MODEL_ID = "model/base_model/llava-onevision-qwen2-7b-si-hf"
BASE = "model/base_model/llava-onevision-qwen2-7b-si-hf"
ORIGINAL = "model/original_model/llava-onevision-qwen2-7b-original"
MFA = "model/unlearned_model/llava-onevision/split_ratio_5/MFA_unsculpted/sculpted"
DATA = "data/S-MLLMUn-data/eval_data"
EVAL_ROOT = "eval_result/llava-onevision/split_ratio_5"
LOCK = threading.RLock()


class GPUUnavailable(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def save_status(status):
    with LOCK:
        temporary = RUN / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(RUN / "status.json")


def load_scoring_credentials():
    """Load the already-configured Conda credential without printing its value."""
    if os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return "inherited environment"
    hook = PYTHON.parent.parent / "etc/conda/activate.d/smfa_api_key.sh"
    if not hook.is_file():
        return "not configured"
    code = ('import json,os; print(json.dumps({k:os.environ[k] for k in '
            '("DASHSCOPE_API_KEY","OPENAI_API_KEY") if os.environ.get(k)}))')
    result = subprocess.run(["bash", "-c", 'set +x; source "$1"; "$2" -c "$3"',
                             "wp1-credential", str(hook), str(PYTHON), code],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Conda scoring credential hook failed; credential values were not logged")
    os.environ.update(json.loads(result.stdout.splitlines()[-1]))
    return str(hook)


def configure_network_environment():
    """Use the verified direct DashScope route in this runner and its children."""
    removed = []
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if os.environ.pop(key, None) is not None:
            removed.append(key)
    return {"route": "direct", "proxy_variables_cleared_in_runner_only": removed,
            "reason": "Inherited socks5h proxy URL is unsupported by the installed experiment HTTP client"}


def preflight(status, min_free_mib):
    if not PYTHON.is_file():
        raise RuntimeError(f"Missing experiment environment Python: {PYTHON}")
    probe = subprocess.run(
        [str(PYTHON), "-c",
         'import sys,json,torch; sys.path.insert(0,"S-MLLMUn"); import eval_fact,eval_meaningful; '
         'print(json.dumps({"torch":torch.__version__,"built_cuda":torch.version.cuda,'
         '"cuda_available":torch.cuda.is_available(),"device_count":torch.cuda.device_count(),'
         '"factuality_client":eval_fact.client is not None,"meaningful_client":eval_meaningful.client is not None}))'],
        cwd=ROOT, capture_output=True, text=True,
    )
    if probe.returncode:
        raise RuntimeError("Experiment Python/import preflight failed: " + probe.stderr[-2000:])
    check = status["preflight"]
    check["python"] = json.loads(probe.stdout.splitlines()[-1])
    check["scoring_configured"] = check["python"]["factuality_client"] and check["python"]["meaningful_client"]
    check["api_connectivity"] = "not probed; configured client does not prove API connectivity"
    check["minimum_free_mib_per_gpu"] = min_free_mib
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.free,memory.total",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if result.returncode:
        check["nvidia_smi_error"] = result.stderr.strip() or result.stdout.strip()
        raise GPUUnavailable("This execution session cannot access NVIDIA GPU devices")
    devices = {}
    for line in result.stdout.splitlines():
        index, uuid, name, free, total = [item.strip() for item in line.split(",")]
        devices[int(index)] = {"index": int(index), "uuid": uuid, "name": name,
                               "free_mib": int(free), "total_mib": int(total)}
    check["gpus"] = list(devices.values())
    if not check["python"]["cuda_available"] or check["python"]["device_count"] < 2:
        raise GPUUnavailable("PyTorch must see two CUDA GPUs in the launching session")
    reasons = []
    for index in (0, 1):
        if index not in devices:
            reasons.append(f"GPU {index} is unavailable")
        elif devices[index]["free_mib"] < min_free_mib:
            reasons.append(f"GPU {index} has only {devices[index]['free_mib']} MiB free")
    if reasons:
        raise GPUUnavailable("; ".join(reasons))


def parse_zero_masks(log_text):
    matches = re.findall(r"^\[(multi|text)\]\s+masked parameters:\s*(\d+)/(\d+)", log_text, re.M)
    if len(matches) != 2 or {name for name, _, _ in matches} != {"multi", "text"}:
        raise ValueError("Official sculpt log must contain one mask count for each adapter")
    counts = {name: {"masked": int(masked), "total": int(total)} for name, masked, total in matches}
    if any(value["masked"] != 0 or value["total"] <= 0 for value in counts.values()):
        raise ValueError("MFA baseline must have zero masks and a positive parameter total for both adapters")
    return counts


def check_saved_checkpoint():
    folder = REPO / MFA
    index = json.loads((folder / "model.safetensors.index.json").read_text())
    if not index["weight_map"] or not (folder / "config.json").is_file():
        raise RuntimeError("Official MFA checkpoint is incomplete")
    for name in set(index["weight_map"].values()):
        if not (folder / name).is_file():
            raise RuntimeError(f"Missing MFA shard: {name}")


def run_stage(status, label, stage, command, gpu, log_path, check_zero_mask=False):
    completed = [r for r in status[label]["stages"] if r["stage"] == stage and r["state"] == "complete"]
    if completed:
        if completed[-1]["command"] != command:
            raise RuntimeError(f"Cannot reuse {label}/{stage}: command changed")
        if check_zero_mask:
            parse_zero_masks(Path(completed[-1]["log"]).read_text())
            check_saved_checkpoint()
        return
    attempt = 1 + sum(r["stage"] == stage for r in status[label]["stages"])
    log_path = log_path.with_name(f"{label}_{stage}_{attempt}.log")
    record = {"stage": stage, "gpu": gpu, "started_at": now(), "state": "running",
              "command": command, "attempt": attempt, "log": str(log_path)}
    status[label]["state"] = "running"
    status[label]["stages"].append(record)
    save_status(status)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4")
    start = time.monotonic()
    print(f"Start {label}/{stage} on GPU {gpu}: {log_path}", flush=True)
    with log_path.open("w") as log:
        process = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    record["duration_seconds"] = round(time.monotonic() - start, 3)
    record["ended_at"] = now()
    if process.returncode:
        record.update(state="failed", exit_code=process.returncode)
        save_status(status)
        raise RuntimeError(f"{label}/{stage} failed with exit code {process.returncode}; see {log_path}")
    if check_zero_mask:
        log_text = log_path.read_text(errors="replace")
        try:
            counts = parse_zero_masks(log_text)
            check_saved_checkpoint()
        except Exception as error:
            record.update(state="failed", error=str(error))
            save_status(status)
            raise
        record["masked_parameters"] = counts
        evidence = {"state": "passed", "checked_at": now(), "multi_k": 1e20, "text_k": 1e20,
                    "epoch": 3, "counts": counts, "log": str(log_path),
                    "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
                    "checkpoint": str(REPO / MFA),
                    "official_script_sha256": hashlib.sha256((REPO / "unlearning_methods/SMFA_sculpt.py").read_bytes()).hexdigest()}
        (RUN / "MFA_unsculpted_zero_mask.json").write_text(json.dumps(evidence, indent=2) + "\n")
    record["state"] = "complete"
    save_status(status)


def evaluate_command(label, checkpoint):
    return [str(PYTHON), "-u", str(ROOT / "reproduce_smfa/evaluate.py"),
            "--model_id", MODEL_ID, "--cache_path", checkpoint,
            "--data_split_folder", DATA, "--output_file", label,
            "--output_folder", f"{EVAL_ROOT}/{label}", "--forget_ratio", "5"]


def worker(status, label, gpu):
    log_path = RUN / f"{label}.log"
    if label == "Base":
        # Check saved SMFA independently while GPU 1 starts official MFA sculpting.
        try:
            run_stage(status, "SMFA_answer_check", "verify_answers", [str(PYTHON), "-u",
                      str(ROOT / "wp1_review/memory_construct.py"), "--verify-answers"],
                      gpu, RUN / "SMFA_answer_check.log")
            status["SMFA_answer_check"]["state"] = "complete"
        except Exception as error:
            status["SMFA_answer_check"]["state"] = "failed"
            status["errors"].append({"model": "SMFA_answer_check", "error": str(error)})
        run_stage(status, label, "evaluate", evaluate_command(label, BASE), gpu, log_path)
        if not status["preflight"]["scoring_configured"]:
            raise RuntimeError("Base generation is complete, but scoring credentials are unavailable")
        run_stage(status, label, "score", [str(PYTHON), "-u", str(ROOT / "reproduce_smfa/score.py"),
                  "--input_folder", f"{EVAL_ROOT}/{label}", "--workers", "4"], gpu, log_path)
    else:
        sculpt_command = [str(PYTHON), "-u", str(REPO / "unlearning_methods/SMFA_sculpt.py"),
                          "--model_id", MODEL_ID, "--original_dir", ORIGINAL,
                          "--save_dir", MFA,
                          "--MFA_multi", "model/unlearned_model/llava-onevision/split_ratio_5/SMFA/epoch_3/MFA_multi",
                          "--MFA_text", "model/unlearned_model/llava-onevision/split_ratio_5/SMFA/epoch_3/MFA_text",
                          "--RA", "model/unlearned_model/llava-onevision/split_ratio_5/SMFA/epoch_3/RA",
                          "--multi_k", "1e20", "--text_k", "1e20"]
        run_stage(status, label, "sculpt", sculpt_command, gpu, log_path, check_zero_mask=True)
        run_stage(status, label, "evaluate", evaluate_command("MFA_unsculpted", MFA), gpu, log_path)
        if not status["preflight"]["scoring_configured"]:
            raise RuntimeError("MFA generation is complete, but scoring credentials are unavailable")
        run_stage(status, label, "score", [str(PYTHON), "-u", str(ROOT / "reproduce_smfa/score.py"),
                  "--input_folder", f"{EVAL_ROOT}/MFA_unsculpted", "--workers", "4"], gpu, log_path)
    status[label]["state"] = "complete"
    save_status(status)


def write_timings(status):
    summary = {"state": status["state"], "models": {},
               "scope": "Wall seconds summed over recorded attempts. Evaluation includes loading/ROUGE; scoring includes API waits/retries. Sculpture and SMFA answer check are separate."}
    for label in ("Base", "MFA_unsculpted"):
        model = {"state": status[label]["state"]}
        for stage in ("sculpt", "evaluate", "score"):
            records = [r for r in status[label]["stages"] if r["stage"] == stage]
            model[stage] = {"state": records[-1]["state"] if records else "not_started",
                            "duration_seconds": round(sum(r.get("duration_seconds", 0) for r in records), 3) if records else None,
                            "attempts": len(records), "incomplete_timing": any("duration_seconds" not in r for r in records)}
        summary["models"][label] = model
    (RUN / "timings.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait-for-gpus", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=32768)
    args = parser.parse_args()
    if not args.launch:
        parser.error("Pass --launch to start the two-GPU pipelines")
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / "runner.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A step-0 baseline runner is already active")
    paths = [ROOT / "reproduce_smfa/evaluate.py", ROOT / "reproduce_smfa/score.py",
             REPO / "unlearning_methods/SMFA_sculpt.py", REPO / "eval.py",
             REPO / "eval_fact.py", REPO / "eval_meaningful.py",
             ROOT / "wp1_review/memory_construct.py", ROOT / "wp1_review/fixed_person_split.json",
             ROOT / "wp1_review/prepare_wp1.py", Path(__file__).resolve()]
    fingerprints = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    previous_path = RUN / "status.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    has_work = previous is not None and any(previous.get(label, {}).get("stages")
                                             for label in ("Base", "MFA_unsculpted", "SMFA_answer_check"))
    if has_work and not args.resume:
        raise SystemExit("Existing stage evidence found; use --resume to retain its timings")
    if has_work and previous.get("source_sha256") != fingerprints:
        raise SystemExit("Experiment sources or split changed since launch; inspect before resuming")
    if previous:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        (RUN / f"status_before_{stamp}.json").write_text(json.dumps(previous, indent=2) + "\n")
    status = {"state": "running", "started_at": now(), "assignment": {"Base": 0, "MFA_unsculpted": 1},
              "models": "Base checkpoint and official SMFA sculptor with k_multi=k_text=1e20",
              "scoring": "reproduce_smfa/score.py, factuality and meaningfulness, 4 workers",
              "preflight": {"state": "running", "started_at": now()},
              "Base": {"state": "not_started", "stages": []},
              "MFA_unsculpted": {"state": "not_started", "stages": []},
              "SMFA_answer_check": {"state": "not_started", "stages": []}}
    if has_work:
        for label in ("Base", "MFA_unsculpted", "SMFA_answer_check"):
            status[label] = previous[label]
        status["started_at"] = previous["started_at"]
    status.update(source_sha256=fingerprints, runner_pid=os.getpid(), errors=[])
    status["network"] = configure_network_environment()
    status["credential_source"] = load_scoring_credentials()
    save_status(status)
    while True:
        try:
            preflight(status, args.min_free_mib)
            break
        except Exception as error:
            waiting = args.wait_for_gpus and isinstance(error, GPUUnavailable)
            state = "waiting_for_gpus" if waiting else "blocked"
            status["preflight"].update(state=state, checked_at=now(), error=str(error))
            status["state"] = state
            for label in ("Base", "MFA_unsculpted", "SMFA_answer_check"):
                if status[label]["state"] != "complete":
                    status[label]["state"] = state
            save_status(status)
            write_timings(status)
            print(str(error), flush=True)
            if not waiting:
                raise SystemExit(2)
            time.sleep(30)
    sculpt_done = any(r["stage"] == "sculpt" and r["state"] == "complete"
                      for r in status["MFA_unsculpted"]["stages"])
    if (REPO / MFA).exists() and not sculpt_done:
        status.update(state="blocked", error=f"MFA output exists without completed official sculpt evidence: {REPO / MFA}")
        save_status(status)
        write_timings(status)
        raise SystemExit(2)
    status["state"] = "running"
    status["preflight"].pop("error", None)
    status["preflight"].update(state="complete", ended_at=now())
    save_status(status)
    errors = status["errors"]
    def guarded(label, gpu):
        try:
            worker(status, label, gpu)
        except BaseException as error:
            status[label]["state"] = "failed"
            errors.append({"model": label, "error": str(error)})
            save_status(status)
    threads = [threading.Thread(target=guarded, args=("Base", 0)),
               threading.Thread(target=guarded, args=("MFA_unsculpted", 1))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if any(status[label]["state"] == "complete" for label in ("Base", "MFA_unsculpted")):
        result = subprocess.run([str(PYTHON), str(ROOT / "wp1_review/prepare_wp1.py"), "--analyze-existing"], cwd=ROOT)
        status["postprocess_exit_code"] = result.returncode
        if result.returncode:
            errors.append({"stage": "postprocess", "error": "Split analysis failed"})
    status["ended_at"] = now()
    status["state"] = "failed" if errors else "complete"
    status["errors"] = errors
    save_status(status)
    write_timings(status)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
