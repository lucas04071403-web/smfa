"""Reconstruct every formal epoch-3 sculpted LoRA layer independently."""

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import audit_final as auditor
import checkpoint_names

ROOT, RUN, REPO = auditor.ROOT, auditor.RUN, auditor.REPO
NAMES = ("MFA_multi", "MFA_text", "RA")


def reference_weight(base, matrices, scales, device="cuda"):
    import torch

    deltas = {}
    for name in NAMES:
        a, b = matrices[name]
        deltas[name] = (scales[name] * (b.to(device) @ a.to(device))).cpu()
    retain = deltas["RA"]
    counts = {}
    masked = {}
    for name, threshold in (("MFA_multi", 5), ("MFA_text", 10)):
        forget = deltas[name]
        norm_ratio = torch.linalg.vector_norm(forget) / (torch.linalg.vector_norm(retain) + 1e-12)
        conflicting = retain.mul(forget).lt(0)
        smaller_retention = (threshold * norm_ratio * retain.abs()).lt(forget.abs())
        removed = conflicting.logical_and(smaller_retention)
        masked[name] = torch.where(removed, torch.zeros_like(forget), forget)
        counts[name] = {"masked": int(removed.sum()), "total": removed.numel()}
    update = (masked["MFA_multi"] + masked["MFA_text"]) / 2
    # The released loader casts the Original model to FP16 before applying the FP32 delta.
    expected = (base.to(device=device, dtype=torch.float16) + update.to(device)).to(torch.float16).cpu()
    return expected, counts


def gpu_zero_users():
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits",
    ], text=True).splitlines()
    uuid = next(row.split(",")[1].strip() for row in rows if row.split(",")[0].strip() == "0")
    users = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], text=True).splitlines()
    return [int(row.split(",")[1]) for row in users if row.split(",")[0].strip() == uuid]


def audit():
    import torch
    from safetensors import safe_open

    torch.set_num_threads(4)
    checkpoint_names.verify_installed_mapping()
    auditor.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Use the idle physical GPU 0")
    auditor.require(not torch.backends.cuda.matmul.allow_tf32, "TF32 differs from the formal sculptor")
    manifest = auditor.read_json(RUN / "manifest.json")
    auditor.require(sys.prefix == manifest["conda_prefix"], "Activate the recorded Conda environment")
    for relative, expected in manifest["source_sha256"].items():
        auditor.require(auditor.sha256(ROOT / relative) == expected, f"Experiment source changed: {relative}")
    pipeline = auditor.read_json(RUN / "status.json")
    auditor.require("sculpt" in pipeline["completed_stages"], "Formal sculpting has not completed")
    checkpoint = auditor.CHECKPOINTS
    sculpted = checkpoint / "sculpted"
    original = REPO / "model/original_model/llava-onevision-qwen2-7b-original"
    source_index = auditor.read_json(original / "model.safetensors.index.json")["weight_map"]
    target_index = auditor.read_json(sculpted / "model.safetensors.index.json")["weight_map"]
    auditor.require(set(source_index) == set(target_index), "Checkpoint tensor names differ")
    stats = auditor.read_json(sculpted / "sculpt_stats.json")
    auditor.require((stats["multi_k"], stats["text_k"]) == (5, 10), "Sculpting thresholds differ")
    epoch_audit = auditor.read_json(RUN / "epoch_3_audit.json")
    recorded_adapters = {item["adapter"]: item for item in epoch_audit["adapters"]}
    adapters, scales, hashes = {}, {}, {}
    for name in NAMES:
        folder = checkpoint / "epoch_3" / name
        path = folder / "adapter_model.safetensors"
        config = auditor.read_json(folder / "adapter_config.json")
        expected = recorded_adapters[name]
        auditor.require(auditor.sha256(path) == expected["sha256"], "Epoch-3 adapter changed")
        auditor.require(auditor.sha256(folder / "adapter_config.json") == expected["config_sha256"],
                        "Epoch-3 adapter configuration changed")
        auditor.require((config["r"], config["lora_alpha"]) == (16, 16)
                        and not any(config.get(key) for key in (
                            "fan_in_fan_out", "use_dora", "use_rslora", "rank_pattern", "alpha_pattern")),
                        "Unsupported adapter configuration")
        with safe_open(path, framework="pt", device="cpu") as handle:
            adapters[name] = {key: handle.get_tensor(key) for key in handle.keys()}
        auditor.require(all(tensor.dtype == torch.float32 for tensor in adapters[name].values()),
                        "Formal LoRA tensors are not FP32")
        scales[name] = config["lora_alpha"] / config["r"]
        hashes[str(path.relative_to(ROOT))] = expected["sha256"]
    keys = set(adapters["RA"])
    auditor.require(all(set(adapters[name]) == keys for name in NAMES), "Adapter tensor names differ")
    layers = sorted(key.removesuffix(".lora_A.weight") for key in keys if key.endswith(".lora_A.weight"))
    auditor.require(len(layers) == stats["layers"] == 274, "Adapted layer coverage differs")
    totals = {name: {"masked": 0, "total": 0} for name in NAMES[:2]}
    inspected = []
    with ExitStack() as stack:
        originals = {shard: stack.enter_context(safe_open(original / shard, framework="pt", device="cpu"))
                     for shard in set(source_index.values())}
        targets = {shard: stack.enter_context(safe_open(sculpted / shard, framework="pt", device="cpu"))
                   for shard in set(target_index.values())}
        with torch.no_grad():
            for number, layer in enumerate(layers, 1):
                runtime_key = layer.removeprefix("base_model.model.") + ".weight"
                key = checkpoint_names.runtime_to_checkpoint(runtime_key)
                auditor.require(key in source_index and key in target_index, f"Missing adapted layer: {key}")
                base = originals[source_index[key]].get_tensor(key)
                actual = targets[target_index[key]].get_tensor(key)
                auditor.require(actual.dtype == torch.float16, "Sculpted weight is not FP16")
                matrices = {name: (adapters[name][layer + ".lora_A.weight"],
                                   adapters[name][layer + ".lora_B.weight"]) for name in NAMES}
                expected, counts = reference_weight(base, matrices, scales)
                if not torch.equal(expected, actual):
                    difference = (expected.float() - actual.float()).abs()
                    auditor.write_json(RUN / "sculpt_formula_mismatch.json", {
                        "state": "mismatch", "layer": key, "checked_at": auditor.now(),
                        "different_elements": int(expected.ne(actual).sum()),
                        "maximum_absolute_difference": float(difference.max()),
                    })
                    raise ValueError(f"Sculpted layer differs from independent reconstruction: {key}")
                for name in totals:
                    for field in ("masked", "total"):
                        totals[name][field] += counts[name][field]
                inspected.append({"layer": key, "runtime_layer": runtime_key,
                                  "elements": actual.numel(), "exact_equal": True})
                if number % 28 == 0 or number == len(layers):
                    print(f"Exactly reconstructed {number}/{len(layers)} sculpted layers", flush=True)
                del base, actual, expected, counts, matrices
    for name, suffix in (("MFA_multi", "multi"), ("MFA_text", "text")):
        auditor.require(totals[name] == {key: stats[suffix][key] for key in ("masked", "total")},
                        f"Actual mask counts differ from reconstruction: {suffix}")
    for folder, index in ((original, source_index), (sculpted, target_index)):
        for name in sorted(set(index.values()) | {"model.safetensors.index.json"}):
            hashes[str((folder / name).relative_to(ROOT))] = auditor.sha256(folder / name)
    hashes[str((sculpted / "sculpt_stats.json").relative_to(ROOT))] = auditor.sha256(sculpted / "sculpt_stats.json")
    return {"state": "passed", "checked_at": auditor.now(), "scope": "Independent actual epoch-3 sculpting formula check",
            "layers": inspected, "exactly_reconstructed_layers": len(inspected), "mask_counts": totals,
            "multi_k": 5, "text_k": 10, "all_adapted_weights_exact_equal": True,
            "physical_gpu": 0, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cpu_threads": torch.get_num_threads(), "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "checkpoint_conversion_mapping": checkpoint_names.CONVERSION,
            "key_mapping_sha256": auditor.sha256(Path(checkpoint_names.__file__)),
            "sha256": hashes, "source_sha256": manifest["source_sha256"],
            "auditor_sha256": auditor.sha256(Path(__file__)), "api_calls": 0,
            "note": "Other tensor integrity and complete evaluation coverage remain the separate final auditor's responsibility."}


def main(watch):
    os.umask(0o077)
    with (RUN / "sculpt_formula_audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        own_hash = auditor.sha256(Path(__file__))
        naming_hash = auditor.sha256(Path(checkpoint_names.__file__))
        state = {"pid": os.getpid(), "state": "running", "started_at": auditor.now(),
                 "auditor_sha256": own_hash, "key_mapping_sha256": naming_hash,
                 "stage": "waiting_for_formal_sculpt"}
        status = RUN / "sculpt_formula_audit_status.json"
        auditor.write_json(status, state)
        try:
            if watch:
                while True:
                    pipeline = auditor.read_json(RUN / "status.json")
                    auditor.require(pipeline["state"] in ("running", "complete"), "Formal pipeline stopped")
                    auditor.require(pipeline["state"] == "complete" or auditor.live(pipeline["pid"]),
                                    "Formal pipeline controller exited")
                    if "sculpt" in pipeline["completed_stages"]:
                        users = gpu_zero_users()
                        if not users:
                            break
                        state.update(stage="waiting_for_idle_gpu_0", gpu_0_pids=users)
                        auditor.write_json(status, state)
                    time.sleep(30)
            auditor.require(auditor.sha256(Path(__file__)) == own_hash, "Audit source changed after launch")
            auditor.require(auditor.sha256(Path(checkpoint_names.__file__)) == naming_hash,
                            "Checkpoint naming source changed after launch")
            state.update(stage="reconstructing_actual_layers", checked_at=auditor.now())
            auditor.write_json(status, state)
            result = audit()
            destination = RUN / "sculpt_formula_audit.json"
            auditor.require(not destination.exists(), "Refusing to replace an existing formula audit")
            auditor.write_json(destination, result)
            state.update(state="complete", stage="complete", completed_at=auditor.now(),
                         audit_sha256=auditor.sha256(destination))
            auditor.write_json(status, state)
        except BaseException as error:
            state.update(state="failed", error=str(error), failed_at=auditor.now())
            auditor.write_json(status, state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        os.umask(0o077)
        with (RUN / "sculpt_formula_audit.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--watch"],
                                       env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Launched independent sculpting formula watcher PID {process.pid}")
    else:
        main(args.watch)
