"""Audit the remaining complete training epochs after their checkpoints close."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import audit_final as auditor

RUN = auditor.RUN
STREAMS = {"MFA Multi": ("MFA_multi", 600), "MFA Text": ("MFA_text", 500),
           "RA Multi": ("RA_multi", 300), "RA Text": ("RA_text", 250)}


def training_evidence(epoch, lines):
    progress = re.compile(rf"^Epoch {epoch} - (MFA Multi|MFA Text|RA Multi|RA Text):.*?\b(\d+)/(\d+)")
    loss = re.compile(rf"^Epoch {epoch} - (MFA Multi|MFA Text|RA Multi|RA Text) Loss: (\S+)$")
    completed, losses, evidence = {}, {}, {}
    for line in lines:
        match = progress.search(line)
        if match:
            name, done, total = match.groups()
            key, expected = STREAMS[name]
            auditor.require(int(total) == expected and 0 <= int(done) <= expected,
                            f"Unexpected epoch {epoch} {name} batch coverage")
            if int(done) == expected:
                completed[key] = {"completed_batches": expected, "expected_batches": expected}
                evidence[key + "_completed"] = line
        match = loss.search(line)
        if match:
            name, value = match.groups()
            key, _ = STREAMS[name]
            number = float(value)
            auditor.require(math.isfinite(number), f"Nonfinite epoch {epoch} {name} final loss")
            losses[key] = number
            evidence[key + "_loss"] = line
    expected_keys = {key for key, _ in STREAMS.values()}
    auditor.require(set(completed) == set(losses) == expected_keys,
                    f"Epoch {epoch} lacks all four complete streams and finite final losses")
    return completed, losses, evidence


def checkpoint_closed(epoch, pipeline, text):
    if "train" in pipeline["completed_stages"]:
        return True
    return epoch < 5 and f"Epoch {epoch + 1} - MFA Multi:" in text


def audit_epoch(epoch):
    import torch
    from safetensors import safe_open

    torch.set_num_threads(2)
    manifest = auditor.read_json(RUN / "manifest.json")
    auditor.require(sys.prefix == manifest["conda_prefix"], "Activate the recorded Conda environment")
    for relative, expected in manifest["source_sha256"].items():
        auditor.require(auditor.sha256(auditor.ROOT / relative) == expected, f"Experiment source changed: {relative}")
    text = (RUN / "train.log").read_text(errors="replace")
    auditor.require(checkpoint_closed(epoch, auditor.read_json(RUN / "status.json"), text),
                    "The requested epoch may still be writing its checkpoints")
    streams, losses, evidence = training_evidence(epoch, text.splitlines())
    prior = auditor.read_json(RUN / f"epoch_{epoch - 1}_audit.json")
    prior_records = {item["adapter"]: item for item in prior["adapters"]}
    records = []
    for name in ("MFA_multi", "MFA_text", "RA"):
        folder = auditor.CHECKPOINTS / f"epoch_{epoch}" / name
        path = folder / "adapter_model.safetensors"
        previous = auditor.CHECKPOINTS / f"epoch_{epoch - 1}" / name / path.name
        config = auditor.read_json(folder / "adapter_config.json")
        auditor.require((config["r"], config["lora_alpha"], config["lora_dropout"]) == (16, 16, 0.05),
                        "LoRA configuration differs from the released trainer")
        auditor.require(auditor.sha256(previous) == prior_records[name]["sha256"], "Previous audited checkpoint changed")
        path_hash = auditor.sha256(path)
        with safe_open(path, framework="pt", device="cpu") as current, \
                safe_open(previous, framework="pt", device="cpu") as old:
            keys = set(current.keys())
            auditor.require(len(keys) == 548 and keys == set(old.keys()), "Adapter tensor coverage differs")
            parameters = 0
            language, vision, changed = [], [], 0
            for key in sorted(keys):
                tensor, previous_tensor = current.get_tensor(key), old.get_tensor(key)
                auditor.require(tensor.shape == previous_tensor.shape and bool(torch.isfinite(tensor).all()),
                                f"Invalid learned tensor: {name}/{key}")
                parameters += tensor.numel()
                changed += int(not torch.equal(tensor, previous_tensor))
                if ".lora_B." in key:
                    bucket = vision if "vision_tower" in key else language
                    bucket.append(bool(tensor.count_nonzero()))
            auditor.require(parameters == 43245568 and changed > 0, "Adapter parameters or learned updates differ")
            auditor.require(len(language) == 196 and all(language), "Some language LoRA B tensors are unlearned")
            auditor.require(len(vision) == 78 and (not any(vision) if name == "MFA_text" else all(vision)),
                            "Vision LoRA B learning state differs")
        auditor.require(auditor.sha256(path) == path_hash, "Audited checkpoint changed while reading")
        records.append({"adapter": name, "tensors": len(keys), "parameters": parameters, "all_finite": True,
                        "language_lora_B_tensors": len(language), "nonzero_language_lora_B_tensors": sum(language),
                        "vision_lora_B_tensors": len(vision), "nonzero_vision_lora_B_tensors": sum(vision),
                        "bytes": path.stat().st_size, "sha256": path_hash,
                        "config_sha256": auditor.sha256(folder / "adapter_config.json"),
                        f"different_from_epoch_{epoch - 1}": True,
                        f"changed_tensors_from_epoch_{epoch - 1}": changed})
        print(f"Verified epoch {epoch} {name}: {changed} changed tensors", flush=True)
    return {"epoch": epoch, "state": "all_three_saved_adapters_valid_and_learned", "checked_at": auditor.now(),
            "scope": f"{epoch} complete training epochs; evaluation and final audits remain separate requirements",
            "checkpoint_root": str((auditor.CHECKPOINTS / f"epoch_{epoch}").relative_to(auditor.ROOT)),
            "losses": losses, "training_streams": streams, "adapters": records,
            "log_completion_evidence": evidence,
            "log_evidence_sha256": hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest(),
            "source_sha256": manifest["source_sha256"], "auditor_sha256": auditor.sha256(Path(__file__)),
            "note": "MFA_text vision B tensors remain zero because this stream has no image inputs.",
            "gpu_allocations": 0, "api_calls": 0}


def main():
    os.umask(0o077)
    with (RUN / "training_epoch_audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_hashes = {str(Path(module.__file__)): auditor.sha256(Path(module.__file__))
                         for module in (auditor, sys.modules[__name__])}
        state = {"pid": os.getpid(), "state": "running", "started_at": auditor.now(),
                 "completed_epochs": [], "source_sha256": source_hashes}
        status = RUN / "training_epoch_audit_status.json"
        auditor.write_json(status, state)
        try:
            for epoch in (4, 5):
                state.update(stage=f"waiting_for_epoch_{epoch}")
                auditor.write_json(status, state)
                while True:
                    pipeline = auditor.read_json(RUN / "status.json")
                    auditor.require(pipeline["state"] in ("running", "complete"), "Formal pipeline stopped")
                    auditor.require(pipeline["state"] == "complete" or auditor.live(pipeline["pid"]),
                                    "Formal pipeline controller exited")
                    text = (RUN / "train.log").read_text(errors="replace")
                    if checkpoint_closed(epoch, pipeline, text):
                        break
                    time.sleep(30)
                for path, expected in source_hashes.items():
                    auditor.require(auditor.sha256(Path(path)) == expected, "Training audit source changed")
                destination = RUN / f"epoch_{epoch}_audit.json"
                if destination.exists():
                    existing = auditor.read_json(destination)
                    auditor.require(existing["epoch"] == epoch
                                    and existing["state"] == "all_three_saved_adapters_valid_and_learned",
                                    "Existing epoch audit is not valid")
                    for item in existing["adapters"]:
                        path = auditor.CHECKPOINTS / f"epoch_{epoch}" / item["adapter"] / "adapter_model.safetensors"
                        auditor.require(auditor.sha256(path) == item["sha256"], "Previously audited epoch changed")
                else:
                    state.update(stage=f"auditing_epoch_{epoch}")
                    auditor.write_json(status, state)
                    auditor.write_json(destination, audit_epoch(epoch))
                state["completed_epochs"].append(epoch)
                auditor.write_json(status, state)
            state.update(state="complete", stage="complete", completed_at=auditor.now())
            auditor.write_json(status, state)
        except BaseException as error:
            state.update(state="failed", error=str(error), failed_at=auditor.now())
            auditor.write_json(status, state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        os.umask(0o077)
        with (RUN / "training_epoch_audit.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())],
                                       env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Launched remaining-epoch auditor PID {process.pid}")
    else:
        main()
