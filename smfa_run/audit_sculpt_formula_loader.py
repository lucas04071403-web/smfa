"""Reconstruct actual sculpted weights using the verified PEFT loading precision."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import audit_final as auditor
import audit_sculpt_formula as formula
import checkpoint_names


def validate_loading():
    import peft
    from peft import LoraConfig, PeftModel
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch

    runtime = "base_model.model.model.language_model.layers.0.mlp.down_proj"
    key = checkpoint_names.runtime_to_checkpoint(runtime.removeprefix("base_model.model.") + ".weight")
    matrices = {}
    for name in formula.NAMES:
        path = auditor.CHECKPOINTS / "epoch_3" / name / "adapter_model.safetensors"
        with safe_open(path, framework="pt", device="cpu") as handle:
            matrices[name] = tuple(handle.get_tensor(runtime + f".lora_{kind}.weight") for kind in ("A", "B"))

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            a, b = matrices["MFA_multi"]
            self.dense = torch.nn.Linear(a.shape[1], b.shape[0], bias=False, dtype=torch.float16)

        def forward(self, inputs):
            return self.dense(inputs)

    loaded = {}
    with tempfile.TemporaryDirectory(prefix="smfa_precision_probe_") as temporary:
        paths = {}
        for name, pair in matrices.items():
            path = Path(temporary) / name
            path.mkdir()
            config = LoraConfig.from_pretrained(auditor.CHECKPOINTS / "epoch_3" / name)
            config.task_type = None
            config.target_modules = {"dense"}
            config.save_pretrained(path)
            save_file({f"base_model.model.dense.lora_{kind}.weight": tensor
                       for kind, tensor in zip(("A", "B"), pair)}, path / "adapter_model.safetensors")
            paths[name] = path
        model = PeftModel.from_pretrained(Probe(), paths["MFA_multi"],
                                         adapter_name="MFA_multi", torch_device="cpu")
        model.load_adapter(paths["MFA_text"], adapter_name="MFA_text", torch_device="cpu")
        model.load_adapter(paths["RA"], adapter_name="RA", torch_device="cpu")
        layer = model.base_model.model.dense
        for name in formula.NAMES:
            loaded[name] = (layer.lora_A[name].weight.detach().clone(),
                            layer.lora_B[name].weight.detach().clone())
    policy = {}
    for name in formula.NAMES:
        pair = matrices[name]
        expected = pair if name == "MFA_multi" else tuple(tensor.half().float() for tensor in pair)
        auditor.require(all(tensor.dtype == torch.float32 for tensor in loaded[name]),
                        "The installed loader no longer returns FP32 adapter matrices")
        auditor.require(all(torch.equal(actual, target) for actual, target in zip(loaded[name], expected)),
                        "Actual PEFT loading precision differs from the reconstruction policy")
        auditor.require(all(not torch.equal(tensor, tensor.half().float()) for tensor in pair),
                        "The actual precision probe cannot distinguish loading paths")
        policy[name] = "saved FP32 preserved" if name == "MFA_multi" else "saved FP32 -> FP16 -> FP32"
    original = auditor.REPO / "model/original_model/llava-onevision-qwen2-7b-original"
    sculpted = auditor.CHECKPOINTS / "sculpted"
    source_index = auditor.read_json(original / "model.safetensors.index.json")["weight_map"]
    target_index = auditor.read_json(sculpted / "model.safetensors.index.json")["weight_map"]
    with safe_open(original / source_index[key], framework="pt", device="cpu") as handle:
        base = handle.get_tensor(key)
    with safe_open(sculpted / target_index[key], framework="pt", device="cpu") as handle:
        actual = handle.get_tensor(key)
    scales = {name: 1.0 for name in formula.NAMES}
    raw, _ = formula.reference_weight(base, matrices, scales)
    raw_differences = int(raw.ne(actual).sum())
    auditor.require(raw_differences > 0, "The probe does not reproduce the recorded precision mismatch")
    corrected, _ = formula.reference_weight(base, loaded, scales)
    auditor.require(torch.equal(corrected, actual), "Actual loaded matrices do not reconstruct the first saved layer")
    package = Path(peft.__file__).parent
    sources = ("peft_model.py", "utils/save_and_load.py", "tuners/tuners_utils.py", "tuners/lora/layer.py")
    return {"state": "passed", "peft_version": peft.__version__, "actual_layer": key,
            "actual_installed_loader_exercised": True, "task_head_omitted_from_linear_probe": True,
            "adapter_loading_order": ["MFA_multi", "MFA_text", "RA"], "precision_policy": policy,
            "raw_FP32_formula_different_elements": raw_differences,
            "actual_loader_formula_first_layer_exact_equal": True,
            "installed_peft_source_sha256": {name: auditor.sha256(package / name) for name in sources}}


def main():
    import torch

    os.umask(0o077)
    torch.set_num_threads(4)
    auditor.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Use physical GPU 0")
    manifest = auditor.read_json(auditor.RUN / "manifest.json")
    auditor.require(sys.prefix == manifest["conda_prefix"], "Activate the recorded Conda environment")
    for relative, expected in manifest["source_sha256"].items():
        auditor.require(auditor.sha256(auditor.ROOT / relative) == expected,
                        f"Experiment source changed: {relative}")
    status_path = auditor.RUN / "sculpt_formula_audit_status.json"
    previous = auditor.read_json(status_path)
    auditor.require(previous["state"] == "failed" and not auditor.live(previous["pid"]),
                    "Inspect and close the previous formula audit before retrying")
    auditor.require(previous.get("error") == "Sculpted layer differs from independent reconstruction: "
                    "language_model.model.layers.0.mlp.down_proj.weight", "Unexpected formula audit failure")
    auditor.require(not (auditor.RUN / "sculpt_formula_audit.json").exists(), "Formula audit already exists")
    for sample in range(3):
        rows = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits",
        ], text=True).splitlines()
        row = next(line for line in rows if line.split(",")[0].strip() == "0")
        _, free, utilization = (int(part.strip()) for part in row.split(","))
        auditor.require(free >= 20 * 1024 and utilization <= 10, "GPU 0 has insufficient spare capacity")
        print(f"GPU 0 preflight {sample + 1}: {free} MiB free, {utilization}% utilization", flush=True)
        if sample < 2:
            time.sleep(5)
    torch.cuda.set_per_process_memory_fraction(4 * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
    proof = validate_loading()
    print("Actual PEFT loader and exact first-layer reconstruction verified", flush=True)
    stamp = auditor.now().replace(":", "").replace("+", "_")
    backup = auditor.RUN / f"sculpt_formula_loader_correction_{stamp}"
    backup.mkdir()
    for name in ("sculpt_formula_audit_status.json", "sculpt_formula_mismatch.json", "sculpt_formula_audit.log"):
        path = auditor.RUN / name
        if path.exists():
            shutil.copy2(path, backup / name)
    auditor.write_json(backup / "loader_precision_validation.json", proof)
    original_reference = formula.reference_weight
    original_audit = formula.audit

    def reference_weight(base, matrices, scales, device="cuda"):
        effective = {name: pair if name == "MFA_multi" else tuple(tensor.half().float() for tensor in pair)
                     for name, pair in matrices.items()}
        return original_reference(base, effective, scales, device)

    def actual_audit():
        result = original_audit()
        result.update(loader_precision_validation=proof,
                      loader_precision_correction_evidence=str(backup.relative_to(auditor.ROOT)),
                      actual_formula_entrypoint="smfa_run/audit_sculpt_formula_loader.py",
                      actual_formula_entrypoint_sha256=auditor.sha256(Path(__file__)),
                      torch_allocator_limit_bytes=4 * 1024 ** 3,
                      max_torch_memory_allocated_bytes=torch.cuda.max_memory_allocated(0))
        return result

    formula.reference_weight = reference_weight
    formula.audit = actual_audit
    formula.main(False)
    result_path = auditor.RUN / "sculpt_formula_audit.json"
    result = auditor.read_json(result_path)
    auditor.require(result["state"] == "passed" and result["exactly_reconstructed_layers"] == 274
                    and result["all_adapted_weights_exact_equal"], "The complete exact formula audit did not pass")
    print(json.dumps({"state": "passed", "layers": 274, "all_adapted_weights_exact_equal": True,
                      "mask_counts": result["mask_counts"], "evidence": str(result_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
