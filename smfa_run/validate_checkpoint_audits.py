"""Validate checkpoint naming and both supplemental audits without GPU/API work."""

from collections import Counter
import json
from pathlib import Path
import tempfile

from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from transformers import AutoConfig, LlavaOnevisionForConditionalGeneration

import audit_final as auditor
import audit_final_names as final_entrypoint
import audit_sculpt_formula as formula
import checkpoint_names

ROOT = auditor.ROOT
torch.set_num_threads(4)
checkpoint_names.verify_installed_mapping()
checks = []

base = torch.tensor([[1., 2.], [3., 4.]])
a = torch.zeros(16, 2)
a[:2] = torch.eye(2)
ra, multi, text, zero = (torch.zeros(2, 16) for _ in range(4))
ra[:, :2] = torch.tensor([[.01, 1.], [1., 1.]])
multi[:, :2] = torch.tensor([[-10., .01], [.01, .01]])
text[:, :2] = torch.tensor([[20., .03], [-.02, .01]])
matrices = {"RA": (a, ra), "MFA_multi": (a, multi), "MFA_text": (a, text)}
actual, counts = formula.reference_weight(base, matrices, {name: 1. for name in formula.NAMES}, device="cpu")
assert torch.equal(actual, torch.tensor([[11., 2.02], [2.995, 4.01]], dtype=torch.float16))
assert counts == {"MFA_multi": {"masked": 1, "total": 4}, "MFA_text": {"masked": 0, "total": 4}}
matrices["MFA_text"] = (a, zero)
actual, counts = formula.reference_weight(base, matrices, {name: 1. for name in formula.NAMES}, device="cpu")
assert torch.equal(actual, torch.tensor([[1., 2.005], [3.005, 4.005]], dtype=torch.float16))
assert counts["MFA_text"] == {"masked": 0, "total": 4}
matrices["RA"] = (a, zero)
_, counts = formula.reference_weight(base, matrices, {name: 1. for name in formula.NAMES}, device="cpu")
assert all(value["masked"] == 0 for value in counts.values())
checks.append("Hand-calculated rank-16 FP16 merge, mask selection, zero vision/text delta and zero retention cases")

original = auditor.REPO / "model/original_model/llava-onevision-qwen2-7b-original"
index = auditor.read_json(original / "model.safetensors.index.json")["weight_map"]
headers = {}
for shard in set(index.values()):
    with (original / shard).open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        headers[shard] = json.loads(handle.read(size))
config = AutoConfig.from_pretrained(original, local_files_only=True)
with init_empty_weights():
    meta_model = LlavaOnevisionForConditionalGeneration(config)
runtime_names = set(meta_model.state_dict())
renamed = {name: checkpoint_names.runtime_to_checkpoint(name) for name in runtime_names}
assert len(runtime_names) == len(set(renamed.values())) == len(index) == 765
assert set(renamed.values()) == set(index)
del meta_model
checks.append("All 765 actual runtime tensor names map bijectively to downloaded Original checkpoint names")

inputs = {}
for name in formula.NAMES:
    with safe_open(auditor.CHECKPOINTS / "epoch_3" / name / "adapter_model.safetensors",
                   framework="pt", device="cpu") as handle:
        inputs[name] = {key: handle.get_slice(key).get_shape() for key in handle.keys()}
keys = set(inputs["RA"])
assert len(keys) == 548 and all(set(inputs[name]) == keys for name in formula.NAMES)
layers = [key.removesuffix(".lora_A.weight") for key in keys if key.endswith(".lora_A.weight")]
assert len(layers) == 274
categories, dtypes = Counter(), Counter()
for layer in layers:
    runtime = layer.removeprefix("base_model.model.") + ".weight"
    key = checkpoint_names.runtime_to_checkpoint(runtime)
    assert runtime in runtime_names and key in index
    entry = headers[index[key]][key]
    assert entry["dtype"] == "F32"
    dtypes[entry["dtype"]] += 1
    for name in formula.NAMES:
        shape_a, shape_b = inputs[name][layer + ".lora_A.weight"], inputs[name][layer + ".lora_B.weight"]
        assert shape_a[0] == shape_b[1] == 16
        assert entry["shape"] == [shape_b[0], shape_a[1]]
    categories["vision" if "vision_tower" in key else "language"] += 1
assert dict(categories) == {"language": 196, "vision": 78}
checks.append("All 274 formal epoch-3 rank-16 LoRA layers map to Original weights with exact compatible shapes")

raw_sculpt_check, raw_final = auditor.audit_sculpted, auditor.final_audit
old_checkpoints, old_repo = auditor.CHECKPOINTS, auditor.REPO
try:
    with tempfile.TemporaryDirectory(prefix="smfa_checkpoint_audit_") as temporary:
        temporary = Path(temporary)
        auditor.CHECKPOINTS = temporary / "checkpoints"
        auditor.REPO = temporary / "repo"
        source = auditor.REPO / "model/original_model/llava-onevision-qwen2-7b-original"
        target = auditor.CHECKPOINTS / "sculpted"
        source.mkdir(parents=True)
        target.mkdir(parents=True)
        language = "model.language_model.layers.0.self_attn.q_proj.weight"
        vision = "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"
        saved_language, saved_vision = map(checkpoint_names.runtime_to_checkpoint, (language, vision))
        other = "image_newline"
        baseline = {saved_language: torch.ones(2, 2), saved_vision: torch.ones(2, 2), other: torch.ones(1)}
        result = {saved_language: torch.full((2, 2), 1.5, dtype=torch.float16),
                  saved_vision: torch.full((2, 2), 2., dtype=torch.float16), other: torch.ones(1, dtype=torch.float16)}
        shard = "model-00001-of-00001.safetensors"
        for folder, tensors in ((source, baseline), (target, result)):
            save_file(tensors, folder / shard)
            auditor.write_json(folder / "model.safetensors.index.json", {"weight_map": {key: shard for key in tensors}})
        auditor.write_json(target / "sculpt_stats.json", {
            "layers": 2, "multi_k": 5, "text_k": 10,
            "multi": {"masked": 0, "total": 8, "ratio": 0.},
            "text": {"masked": 0, "total": 8, "ratio": 0.},
        })
        try:
            raw_sculpt_check({language, vision})
        except ValueError as error:
            assert str(error) == "Sculpted checkpoint is missing adapted weights"
        else:
            raise AssertionError("Frozen final auditor did not reproduce the name mismatch")
        final_entrypoint.install_mapping()
        checked = auditor.audit_sculpted({language, vision})
        assert checked["adapted_weights"] == checked["changed_weights"] == 2
        assert checked["runtime_to_checkpoint_adapted_keys"] == {language: saved_language, vision: saved_vision}
        result[other] = torch.full((1,), 9., dtype=torch.float16)
        save_file(result, target / shard)
        try:
            auditor.audit_sculpted({language, vision})
        except ValueError as error:
            assert "An unadapted weight changed" in str(error)
        else:
            raise AssertionError("Unadapted mutation was accepted")
        result[other] = torch.ones(1, dtype=torch.float16)
        result[saved_language][0, 0] = float("nan")
        save_file(result, target / shard)
        try:
            auditor.audit_sculpted({language, vision})
        except ValueError as error:
            assert "Nonfinite sculpted weight" in str(error)
        else:
            raise AssertionError("Nonfinite tensor was accepted")
finally:
    auditor.audit_sculpted, auditor.final_audit = raw_sculpt_check, raw_final
    auditor.CHECKPOINTS, auditor.REPO = old_checkpoints, old_repo
checks.append("Synthetic saved checkpoints reproduce the frozen audit failure and pass with only adapted-name translation")
checks.append("Unadapted mutations and nonfinite values remain rejected after name translation")
assert not torch.cuda.is_initialized()
record = {"state": "passed", "checked_at": auditor.now(), "checks": checks,
          "runtime_tensor_names": 765, "formal_epoch_3_adapted_layers": dict(categories),
          "original_adapted_weight_dtypes": dict(dtypes),
          "formal_sculpted_checkpoint_checked": False, "formal_outputs_modified": False,
          "gpu_allocations": 0, "api_calls": 0,
          "sha256": {str(Path(module.__file__).relative_to(ROOT)): auditor.sha256(Path(module.__file__))
                     for module in (auditor, formula, final_entrypoint, checkpoint_names)},
          "validator_sha256": auditor.sha256(Path(__file__))}
destination = auditor.RUN / "checkpoint_audits_validation.json"
assert not destination.exists()
auditor.write_json(destination, record)
print(json.dumps(record, indent=2))
