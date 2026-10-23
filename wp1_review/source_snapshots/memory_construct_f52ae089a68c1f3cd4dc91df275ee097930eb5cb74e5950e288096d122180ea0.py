"""Read epoch-3 SMFA LoRAs and construct sculpted weights one tensor at a time."""

import argparse
import hashlib
from io import BytesIO
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "S-MLLMUn"
sys.path.insert(0, str(ROOT / "smfa_run"))
sys.path.insert(0, str(ROOT))
import checkpoint_names
from wp1_review.prepare_wp1 import annotate_questions, atomic_json, load_partition


CHECKPOINTS = REPO / "model/unlearned_model/llava-onevision/split_ratio_5/SMFA"
ORIGINAL = REPO / "model/original_model/llava-onevision-qwen2-7b-original"
SCULPTED = CHECKPOINTS / "sculpted"
EVAL_SMFA = REPO / "eval_result/llava-onevision/split_ratio_5/SMFA"
BASE_MODEL = REPO / "model/base_model/llava-onevision-qwen2-7b-si-hf"
ADAPTERS = ("MFA_multi", "MFA_text", "RA")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lora_matrices(epoch=3):
    """Load A/B factors to CPU with PEFT's verified effective precision."""
    matrices = {}
    scales = {}
    for adapter in ADAPTERS:
        directory = CHECKPOINTS / f"epoch_{epoch}" / adapter
        config = json.loads((directory / "adapter_config.json").read_text())
        if config["r"] != config["lora_alpha"]:
            raise ValueError(f"Expected unit LoRA scale, found {adapter}: {config}")
        path = directory / "adapter_model.safetensors"
        with safe_open(path, framework="pt", device="cpu") as handle:
            values = {key: handle.get_tensor(key) for key in handle.keys()}
        matrices[adapter] = {}
        for key, tensor in values.items():
            # The verified PEFT 0.12 loader preserves the first adapter's FP32,
            # while adapters loaded subsequently pass through the FP16 base dtype.
            effective = tensor if adapter == "MFA_multi" else tensor.half().float()
            matrices[adapter][key] = effective
        scales[adapter] = config["lora_alpha"] / config["r"]
    keys = set(matrices["RA"])
    if any(set(matrices[name]) != keys for name in ADAPTERS):
        raise ValueError("Adapter tensor key sets differ")
    return matrices, scales


def sculpt_delta(retain, forget, k):
    ratio = torch.linalg.vector_norm(forget) / (torch.linalg.vector_norm(retain) + 1e-12)
    mask = retain.mul(forget).lt(0)
    mask &= (k * ratio * retain.abs()).lt(forget.abs())
    return torch.where(mask, torch.zeros_like(forget), forget), int(mask.sum())


def construct_weight(base, matrices, scales, stem, k_multi=5, k_text=10):
    """Construct one final FP16 tensor, matching official loader arithmetic."""
    deltas = {}
    for adapter in ADAPTERS:
        deltas[adapter] = scales[adapter] * (
            matrices[adapter][stem + ".lora_B.weight"] @ matrices[adapter][stem + ".lora_A.weight"]
        )
    multi, count_multi = sculpt_delta(deltas["RA"], deltas["MFA_multi"], k_multi)
    text, count_text = sculpt_delta(deltas["RA"], deltas["MFA_text"], k_text)
    update = (multi + text) / 2
    final = (base.to(torch.float16) + update).to(torch.float16)
    return final, {"MFA_multi": count_multi, "MFA_text": count_text}


def layer_stems(matrices):
    return sorted(key.removesuffix(".lora_A.weight") for key in matrices["RA"]
                  if key.endswith(".lora_A.weight"))


def iter_memory_weights(matrices, scales, k_multi=5, k_text=10, include_counts=False):
    """Yield (checkpoint key, constructed tensor); callers need not retain 14 GB."""
    index = json.loads((ORIGINAL / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(index.values()))
    with ExitStack() as stack:
        handles = {name: stack.enter_context(safe_open(ORIGINAL / name, framework="pt", device="cpu"))
                   for name in shards}
        for stem in layer_stems(matrices):
            runtime = stem.removeprefix("base_model.model.") + ".weight"
            key = checkpoint_names.runtime_to_checkpoint(runtime)
            if key not in index:
                raise KeyError(f"LoRA layer is absent from Original checkpoint: {key}")
            base = handles[index[key]].get_tensor(key)
            weight, counts = construct_weight(base, matrices, scales, stem, k_multi, k_text)
            yield (stem, key, weight, counts) if include_counts else (stem, key, weight)


def apply_memory_weights(model, matrices, scales, k_multi=5, k_text=10):
    """Reset adapted tensors from Original, then apply the update without saving."""
    parameters = dict(model.named_parameters())
    applied = 0
    with torch.no_grad():
        for stem, _, weight in iter_memory_weights(matrices, scales, k_multi, k_text):
            name = stem.removeprefix("base_model.model.") + ".weight"
            if name not in parameters:
                raise KeyError(f"Runtime model has no parameter named {name}")
            parameter = parameters[name]
            if tuple(parameter.shape) != tuple(weight.shape):
                raise ValueError(f"Shape mismatch for {name}: {tuple(parameter.shape)} vs {tuple(weight.shape)}")
            parameter.copy_(weight.to(device=parameter.device, dtype=parameter.dtype))
            applied += 1
    return applied


def self_check():
    sys.path.insert(0, str(ROOT / "reproduce_smfa"))
    sys.path.insert(0, str(REPO))
    import evaluate as official_eval
    import pandas as pd

    matrices, scales = load_lora_matrices(epoch=3)
    original_index = json.loads((ORIGINAL / "model.safetensors.index.json").read_text())["weight_map"]
    sculpted_index = json.loads((SCULPTED / "model.safetensors.index.json").read_text())["weight_map"]
    if set(original_index) != set(sculpted_index):
        raise ValueError("Original and saved SMFA tensor indexes differ")
    stats = {name: 0 for name in ADAPTERS[:2]}
    inspected = 0
    adapted_keys = set()
    untouched = 0
    with ExitStack() as stack:
        originals = {name: stack.enter_context(safe_open(ORIGINAL / name, framework="pt", device="cpu"))
                     for name in set(original_index.values())}
        saved = {name: stack.enter_context(safe_open(SCULPTED / name, framework="pt", device="cpu"))
                 for name in set(sculpted_index.values())}
        for stem, key, actual, removed in iter_memory_weights(matrices, scales, include_counts=True):
            target = saved[sculpted_index[key]].get_tensor(key)
            if actual.dtype != torch.float16 or not torch.equal(actual, target):
                raise ValueError(f"Memory construction mismatch: {key}")
            adapted_keys.add(key)
            for adapter in stats:
                stats[adapter] += removed[adapter]
            inspected += 1
            if inspected % 28 == 0:
                print(f"Verified {inspected}/274 reconstructed tensors", flush=True)
        if inspected != 274:
            raise ValueError(f"Expected 274 adapted tensors, found {inspected}")
        for key in sorted(set(original_index) - adapted_keys):
            original = originals[original_index[key]].get_tensor(key)
            target = saved[sculpted_index[key]].get_tensor(key)
            if original.dtype != target.dtype or not torch.equal(original, target):
                raise ValueError(f"Saved SMFA changed an unadapted tensor: {key}")
            untouched += 1
    recorded = json.loads((SCULPTED / "sculpt_stats.json").read_text())
    for adapter, field in (("MFA_multi", "multi"), ("MFA_text", "text")):
        if stats[adapter] != recorded[field]["masked"]:
            raise ValueError(f"Mask count mismatch for {adapter}: {stats[adapter]} vs {recorded[field]['masked']}")
    evaluation_reference = {}
    for split in ("forget", "retain"):
        result_path = EVAL_SMFA / f"{split}_generation_results.json"
        source_path = REPO / f"data/S-MLLMUn-data/eval_data/{split}_5/train-00000-of-00001.parquet"
        questions = json.loads(result_path.read_text())["Generation_Questions"]
        official_eval.verify_questions(questions, pd.read_parquet(source_path))
        if len(questions) < 30 or any(not str(item["generated_answer"]).strip() for item in questions[:30]):
            raise ValueError(f"Invalid first-30 answer reference in {result_path}")
        evaluation_reference[split] = {
            "questions": len(questions),
            "sha256": file_sha256(result_path),
            "first_30_answers_present": True,
        }
    return {
        "state": "passed",
        "layers": inspected,
        "all_constructed_weights_equal_saved_smfa": True,
        "unadapted_tensors_checked": untouched,
        "all_unadapted_weights_equal_original": True,
        "masked_parameters": stats,
        "loader_precision": {"MFA_multi": "FP32 preserved", "MFA_text": "FP32 -> FP16 -> FP32",
                              "RA": "FP32 -> FP16 -> FP32"},
        "artifacts_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in [SCULPTED / "sculpt_stats.json", SCULPTED / "model.safetensors.index.json",
                         ORIGINAL / "model.safetensors.index.json"] + [
                             CHECKPOINTS / "epoch_3" / adapter / filename
                             for adapter in ADAPTERS for filename in ("adapter_config.json", "adapter_model.safetensors")]
        },
        "evaluation_reference": evaluation_reference,
        "candidate_30_answer_match": "separate check: memory_construct.py --verify-answers (F1 development only)",
    }


def official_answer(processor, model, question, image=None):
    if image is None:
        conversation = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=prompt, return_tensors="pt").to("cuda")
    else:
        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}
        ]}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    decoded = processor.decode(outputs[0][2:], skip_special_tokens=True)
    for marker in ("ASSISTANT:", "Answer:", "assistant\n", "assistant:"):
        if marker in decoded:
            return decoded.split(marker)[1].strip()
    return decoded.strip()


def prepare_answer_reference():
    """Select the first 30 F1 questions in source order, never the raw full-set prefix."""
    import pandas as pd
    partition = load_partition()
    f1 = set(partition["forget"]["F1_development"])
    frame = pd.read_parquet(REPO / "data/S-MLLMUn-data/eval_data/forget_5/train-00000-of-00001.parquet")
    result_path = EVAL_SMFA / "forget_generation_results.json"
    questions = json.loads(result_path.read_text())["Generation_Questions"]
    annotated = annotate_questions(questions, frame)
    selected = [(index, pid, question) for index, (question, pid) in enumerate(annotated) if pid in f1][:30]
    if len(selected) != 30:
        raise ValueError("F1 reference has fewer than 30 questions")
    manifest = {"state": "prepared", "cohort": "F1_development", "questions": 30,
                "order": "source parquet order, filtering to frozen F1 before taking 30 questions",
                "split_manifest_sha256": file_sha256(ROOT / "wp1_review/fixed_person_split.json"),
                "smfa_generation_sha256": file_sha256(result_path),
                "reference": [{"generation_index": index, "person_id": pid,
                               "question_type": q["question type"], "question": q["question"]}
                              for index, pid, q in selected]}
    atomic_json(ROOT / "wp1_review/memory_construct_answer_selection.json", manifest)
    return frame[frame["ID"].astype(str).isin(f1)], selected, manifest


def verify_first_30_answers():
    frame, selected, manifest = prepare_answer_reference()
    if not torch.cuda.is_available():
        return {"state": "blocked", "reason": "CUDA is unavailable in this execution session",
                "cohort": "F1_development", "selected_questions": len(selected),
                "generated_questions": 0, "match_rate": None,
                "selection_manifest": "wp1_review/memory_construct_answer_selection.json"}
    from PIL import Image
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    matrices, scales = load_lora_matrices(epoch=3)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        str(ORIGINAL), torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
        local_files_only=True,
    )
    applied = apply_memory_weights(model, matrices, scales)
    if applied != 274:
        raise ValueError(f"Expected 274 in-memory updates, applied {applied}")
    processor = AutoProcessor.from_pretrained(str(BASE_MODEL), local_files_only=True)
    generated = []
    model.eval()
    with torch.inference_mode():
        for _, row in frame.iterrows():
            face = Image.open(BytesIO(row["character_image"]["bytes"]))
            eye = Image.open(BytesIO(row["ophthalmic_image"]["bytes"]))
            for kind, column in (("image_understanding", "image understanding"),
                                 ("image_memory", "image memory")):
                for item in row[column]:
                    if len(generated) == 30:
                        break
                    image_type = item["Image"]
                    image = face if image_type == "character" else eye
                    generated.append({
                        "question type": kind,
                        "image_id": str(row["ID"]),
                        "image_type": image_type,
                        "question": item["Question"],
                        "generated_answer": official_answer(processor, model, item["Question"], image),
                        "ground_truth": item["Answer"],
                    })
                if len(generated) == 30:
                    break
            if len(generated) < 30:
                for item in row["text memory"]:
                    if len(generated) == 30:
                        break
                    generated.append({
                        "question type": "text_memory",
                        "image_id": str(row["ID"]),
                        "image_type": None,
                        "question": item["Question"],
                        "generated_answer": official_answer(processor, model, item["Question"]),
                        "ground_truth": item["Answer"],
                    })
            if len(generated) == 30:
                break
    if len(generated) != 30:
        raise ValueError(f"Generated only {len(generated)} answers")
    comparisons = []
    for actual, (index, pid, expected) in zip(generated, selected):
        if (actual["question type"], actual["question"], actual["ground_truth"], actual["image_id"]) != (
                expected["question type"], expected["question"], expected["ground_truth"], pid):
            raise ValueError(f"Answer-check question mismatch at generation index {index}")
        comparisons.append({"generation_index": index, "person_id": pid,
                            "question_type": actual["question type"], "question": actual["question"],
                            "reference_answer": expected["generated_answer"],
                            "candidate_answer": actual["generated_answer"],
                            "exact_match": actual["generated_answer"] == expected["generated_answer"]})
    exact = sum(item["exact_match"] for item in comparisons)
    result = {"state": "passed" if exact >= 27 else "failed", "matched_answers": exact,
              "total_answers": len(generated), "match_rate": exact / max(len(generated), 1),
              "criterion": "at least 27 of the first 30 exact answer strings",
              "cohort": "F1_development", "in_memory_layers_applied": applied,
              "split_manifest_sha256": manifest["split_manifest_sha256"],
              "smfa_generation_sha256": manifest["smfa_generation_sha256"],
              "comparisons": comparisons}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--verify-answers", action="store_true")
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--prepare-answer-check", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    started = time.monotonic()
    if args.prepare_answer_check:
        _, _, manifest = prepare_answer_reference()
        print(json.dumps({key: value for key, value in manifest.items() if key != "reference"}, indent=2))
        return
    if args.self_check:
        result = self_check()
        result["duration_seconds"] = round(time.monotonic() - started, 3)
        result["implementation_sha256"] = file_sha256(Path(__file__))
        destination = ROOT / "wp1_review/memory_construct_self_check.json"
        atomic_json(destination, result)
        print(json.dumps(result, indent=2))
        return
    if args.verify_answers:
        result = verify_first_30_answers()
        result["duration_seconds"] = round(time.monotonic() - started, 3)
        destination = ROOT / "wp1_review/memory_construct_answer_check.json"
        atomic_json(destination, result)
        print(json.dumps({key: value for key, value in result.items() if key != "comparisons"}, indent=2))
        if result["state"] != "passed":
            raise SystemExit(2 if result["state"] == "blocked" else 1)
        return
    matrices, scales = load_lora_matrices(args.epoch)
    print(json.dumps({"epoch": args.epoch, "layers": len(layer_stems(matrices)),
                      "memory_weights": "streamed one tensor at a time", "lora_scales": scales}, indent=2))


if __name__ == "__main__":
    main()
