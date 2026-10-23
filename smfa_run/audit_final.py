"""Independently audit the completed default experiment without GPU or API use."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "S-MLLMUn"
RUN = ROOT / "smfa_run"
CHECKPOINTS = REPO / "model/unlearned_model/llava-onevision/split_ratio_5/SMFA"
EVALUATION = REPO / "eval_result/llava-onevision/split_ratio_5"
KINDS = ("image_understanding", "image_memory", "text_memory")
COLUMNS = ("image understanding", "image memory", "text memory")
LABELS = ("Image Understanding", "Image Memory", "Text Memory")
ROUGE_LABELS = ("image_understanding", "Image_memory", "Text_memory")
STAGES = ("train", "sculpt", "evaluate_SMFA", "score_SMFA", "evaluate_Original", "score_Original")


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def equal_mean(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual), f"Invalid mean: {label}")
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), f"Mean differs: {label}")


def audit_split(folder, split, rows):
    from rouge_score.rouge_scorer import RougeScorer

    source = folder / f"{split}_generation_results.json"
    questions = read_json(source)["Generation_Questions"]
    expected = [(kind, row, item) for _, row in rows.iterrows()
                for kind, column in zip(KINDS, COLUMNS) for item in row[column]]
    require(len(questions) == len(expected) > 0, f"Question coverage differs: {source}")
    scorer = RougeScorer(["rougeL"], use_stemmer=True)
    counts = Counter()
    rouge_totals = Counter()
    for index, (question, (kind, row, item)) in enumerate(zip(questions, expected)):
        require((question["question type"], question["question"], question["ground_truth"])
                == (kind, item["Question"], item["Answer"]), f"Question identity differs: {source}/{index}")
        answer = question.get("generated_answer")
        require(isinstance(answer, str) and answer.strip(), f"Empty generation: {source}/{index}")
        if kind != "text_memory":
            require(question.get("image_id") == row["ID"]
                    and question.get("image_type") == item["Image"], f"Image identity differs: {source}/{index}")
        counts[kind] += 1
        rouge_totals[kind] += scorer.score(question["ground_truth"], answer)["rougeL"].fmeasure
    require(set(counts) == set(KINDS), f"Missing category: {source}")
    result = {"profiles": len(rows), "questions": len(questions), "category_counts": dict(counts),
              "rougeL": {kind: rouge_totals[kind] / counts[kind] for kind in KINDS},
              "sha256": {source.name: sha256(source)}}
    for suffix, field in (("factuality", "Factuality Score"), ("meaningful", "Meaningful Score")):
        cache = folder / f"{split}_generation_results_{suffix}.jsonl"
        data = cache.read_bytes()
        require(data.endswith(b"\n"), f"Incomplete cache line: {cache}")
        records = {}
        for line in data.splitlines():
            record = json.loads(line)
            index = record["index"]
            require(type(index) is int and 0 <= index < len(questions), f"Invalid cache index: {cache}")
            require(index not in records, f"Duplicate cache index: {cache}/{index}")
            question = questions[index]
            digest = hashlib.sha256(json.dumps(question, sort_keys=True).encode()).hexdigest()
            require(record["digest"] == digest, f"Cache digest differs: {cache}/{index}")
            score = record["result"]
            require(type(score.get(field)) is int and 0 <= score[field] <= 10, f"Invalid score: {cache}/{index}")
            for destination, origin in (("Question", "question"), ("Generated Answer", "generated_answer"),
                                        ("Ground Truth", "ground_truth"), ("Question Type", "question type"),
                                        ("Image_ID", "image_id"), ("image_type", "image_type")):
                require(score.get(destination) == question.get(origin), f"Score identity differs: {cache}/{index}/{destination}")
            require(isinstance(score.get("Justification"), str) and score["Justification"].strip(),
                    f"Missing justification: {cache}/{index}")
            records[index] = score
        require(set(records) == set(range(len(questions))), f"Missing scores: {cache}")
        output_path = folder / f"{split}_generation_results_{suffix}_score.json"
        output = read_json(output_path)
        means = {}
        for kind, label in zip(KINDS, LABELS):
            grouped = [records[index] for index, question in enumerate(questions) if question["question type"] == kind]
            require(output[kind] == grouped, f"Score summary differs from cache: {output_path}/{kind}")
            means[kind] = sum(item[field] for item in grouped) / counts[kind]
            equal_mean(output["Score"][f"Average {label} Score"], means[kind], f"{output_path}/{kind}")
        result[suffix] = means
        result[f"question_weighted_{suffix}"] = sum(records[index][field] for index in records) / len(questions)
        result["sha256"].update({cache.name: sha256(cache), output_path.name: sha256(output_path)})
    return result


def aggregate(splits):
    forget, retain = splits["forget"], splits["retain"]
    result = {}
    for metric in ("rougeL", "factuality"):
        result[f"image_understanding_{metric}"] = (forget[metric][KINDS[0]] + retain[metric][KINDS[0]]) / 2
        result[f"memory_{metric}"] = (retain[metric][KINDS[1]] + retain[metric][KINDS[2]]
                                      - forget[metric][KINDS[1]] - forget[metric][KINDS[2]]) / 2
    count = forget["questions"] + retain["questions"]
    result["meaningful"] = sum(s["question_weighted_meaningful"] * s["questions"] for s in splits.values()) / count
    return result


def audit_adapters():
    import torch
    from safetensors import safe_open

    records = []
    expected_keys = None
    for epoch in range(1, 6):
        for adapter in ("MFA_multi", "MFA_text", "RA"):
            folder = CHECKPOINTS / f"epoch_{epoch}" / adapter
            config = read_json(folder / "adapter_config.json")
            require((config["r"], config["lora_alpha"], config["lora_dropout"]) == (16, 16, 0.05),
                    f"Unexpected LoRA configuration: {folder}")
            path = folder / "adapter_model.safetensors"
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                require(len(keys) == 548, f"Unexpected adapter tensor count: {path}")
                if expected_keys is None:
                    expected_keys = keys
                require(keys == expected_keys, f"Adapter tensor names differ: {path}")
                language, vision = [], []
                parameters = 0
                for key in sorted(keys):
                    tensor = handle.get_tensor(key)
                    require(bool(torch.isfinite(tensor).all()), f"Nonfinite adapter: {path}/{key}")
                    parameters += tensor.numel()
                    if ".lora_B." in key:
                        target = vision if "vision_tower" in key else language
                        target.append(bool(tensor.count_nonzero()))
                require(len(language) == 196 and all(language), f"Missing learned language delta: {path}")
                require(len(vision) == 78 and (not any(vision) if adapter == "MFA_text" else all(vision)),
                        f"Unexpected learned vision delta: {path}")
                require(parameters == 43245568, f"Adapter parameter count differs: {path}")
            records.append({"epoch": epoch, "adapter": adapter, "parameters": parameters, "tensors": len(keys),
                            "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                            "config_sha256": sha256(folder / "adapter_config.json")})
            print(f"Verified epoch {epoch} {adapter}", flush=True)
    return records, {key.removeprefix("base_model.model.").removesuffix(".lora_B.weight") + ".weight"
                     for key in expected_keys if ".lora_B." in key}


def audit_sculpted(adapted_keys):
    import torch
    from safetensors import safe_open

    folder = CHECKPOINTS / "sculpted"
    original = REPO / "model/original_model/llava-onevision-qwen2-7b-original"
    index = read_json(folder / "model.safetensors.index.json")["weight_map"]
    original_index = read_json(original / "model.safetensors.index.json")["weight_map"]
    require(set(index) == set(original_index), "Sculpted checkpoint tensor names differ")
    require(adapted_keys <= set(index), "Sculpted checkpoint is missing adapted weights")
    stats = read_json(folder / "sculpt_stats.json")
    require((stats["multi_k"], stats["text_k"], stats["layers"]) == (5, 10, len(adapted_keys)),
            "Sculpting configuration or layer coverage differs")
    found, changed = set(), set()
    hashes = {}
    for shard in sorted(set(index.values())):
        path = folder / shard
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                require(key not in found and index.get(key) == shard, f"Wrong sculpted tensor shard: {key}")
                tensor = handle.get_tensor(key)
                require(bool(torch.isfinite(tensor).all()), f"Nonfinite sculpted weight: {key}")
                with safe_open(original / original_index[key], framework="pt", device="cpu") as baseline:
                    previous = baseline.get_tensor(key)
                    require(tensor.shape == previous.shape, f"Sculpted tensor shape differs: {key}")
                    different = not torch.equal(tensor, previous.to(tensor.dtype))
                if different:
                    require(key in adapted_keys, f"An unadapted weight changed: {key}")
                    changed.add(key)
                found.add(key)
        hashes[shard] = sha256(path)
        print(f"Verified sculpted shard {shard}", flush=True)
    require(found == set(index), "Sculpted checkpoint has missing tensors")
    require(any("language_model" in key for key in changed) and any("vision_tower" in key for key in changed),
            "Sculpted checkpoint lacks learned language or vision changes")
    for name in ("multi", "text"):
        item = stats[name]
        require(0 <= item["masked"] <= item["total"] and item["total"] > 0, "Invalid sculpting mask counts")
        equal_mean(item["ratio"], item["masked"] / item["total"], f"sculpt/{name}")
    hashes["model.safetensors.index.json"] = sha256(folder / "model.safetensors.index.json")
    hashes["sculpt_stats.json"] = sha256(folder / "sculpt_stats.json")
    return {"tensors": len(found), "adapted_weights": len(adapted_keys), "changed_weights": len(changed),
            "unadapted_weights_match_original_cast": True, "stats": stats, "sha256": hashes}


def final_audit():
    import pandas as pd
    import torch

    torch.set_num_threads(2)
    status = read_json(RUN / "status.json")
    require(status["state"] == "complete" and status["completed_stages"] == list(STAGES),
            "The full experiment pipeline has not completed")
    manifest = read_json(RUN / "manifest.json")
    require(sys.prefix == manifest["conda_prefix"], "Activate the recorded Conda environment")
    for relative, expected in manifest["source_sha256"].items():
        require(sha256(ROOT / relative) == expected, f"Experiment source changed: {relative}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    require(commit == manifest["repo_commit"], "Official repository revision changed")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True),
            "Official repository worktree changed")
    adapters, adapted_keys = audit_adapters()
    sculpted = audit_sculpted(adapted_keys)
    reference = read_json(RUN / "paper_reference.json")
    results = {}
    for label in ("Original", "SMFA"):
        folder = EVALUATION / label
        stored_path = folder / f"{label}_final_evaluation_results.json"
        stored = read_json(stored_path)
        splits = {}
        for split, group, profiles, questions in (("forget", "Forget Set Results", 50, 950),
                                                 ("retain", "Retain Set (shared dataset) Results", 950, 18050)):
            data = REPO / f"data/S-MLLMUn-data/eval_data/{split}_5/train-00000-of-00001.parquet"
            splits[split] = audit_split(folder, split, pd.read_parquet(data))
            require((splits[split]["profiles"], splits[split]["questions"]) == (profiles, questions),
                    f"Incomplete full evaluation: {label}/{split}")
            for kind, rouge_label in zip(KINDS, ROUGE_LABELS):
                equal_mean(stored[group]["generation"][f"Average ROUGE-L ({rouge_label})"],
                           splits[split]["rougeL"][kind], f"{label}/{split}/ROUGE-L/{kind}")
            splits[split]["data_sha256"] = sha256(data)
            print(f"Verified {label}/{split}: {questions} generations and {questions * 2} judge scores", flush=True)
        overall = aggregate(splits)
        paper_overall = reference[label]["overall"]
        results[label] = {"splits": splits, "overall": overall, "paper_overall": paper_overall,
                          "difference_from_paper": {key: overall[key] - paper_overall[key] for key in overall},
                          "rouge_summary_sha256": sha256(stored_path)}
    return {"state": "passed", "checked_at": now(), "repo_commit": commit,
            "manifest_sha256": sha256(RUN / "manifest.json"), "adapters": adapters, "sculpted": sculpted,
            "generated_answers_verified": 38000, "judge_scores_verified": 76000,
            "results": results, "comparison_limitations": reference["comparison_limitations"],
            "scope": "README default LLaVA-OneVision SMFA, 5% split, five epochs, epoch-3 sculpting; Original comparison"}


def write_report(audit):
    lines = ["# SMFA Final Independent Audit", "", audit["scope"], "",
             "All 15 saved adapters, sculpted checkpoint shards, 38,000 generated answers and",
             "76,000 judge scores passed coverage, identity, range, finiteness and aggregation checks.", "",
             "| Model | Metric | Measured | Paper | Difference |", "| --- | --- | --- | --- | --- |"]
    for label, result in audit["results"].items():
        for metric, value in result["overall"].items():
            lines.append(f"| {label} | {metric} | {value:.6f} | {result['paper_overall'][metric]:.6f} | "
                         f"{result['difference_from_paper'][metric]:+.6f} |")
    lines += ["", "Overall image understanding averages forget and retain category means.",
              "Memory is (retain IM + retain TM - forget IM - forget TM) / 2.",
              "Meaningfulness averages all question scores across both complete splits.", ""]
    lines += [f"- {item}" for item in audit["comparison_limitations"]]
    lines += ["", "Passing this audit verifies execution and result integrity; the table records numerical",
              "agreement or differences with the paper. This does not reproduce its other baselines or ablations.",
              "Machine-readable evidence: `smfa_run/final_audit.json`."]
    destination = ROOT / "SMFA_FINAL_AUDIT.md"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(destination)


def live(pid):
    try:
        return not Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].startswith("Z ")
    except FileNotFoundError:
        return False


def main(watch):
    os.umask(0o077)
    with (RUN / "final_audit.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        own_hash = sha256(Path(__file__))
        state = {"state": "running", "pid": os.getpid(), "started_at": now(), "launcher_sha256": own_hash}
        status_path = RUN / "final_audit_status.json"
        write_json(status_path, state)
        try:
            if watch:
                previous = None
                while True:
                    status = read_json(RUN / "status.json")
                    if status["state"] == "complete":
                        break
                    require(status["state"] == "running", f"Pipeline stopped: {status['state']}")
                    require(live(status["pid"]), "Pipeline controller exited before completing")
                    if status.get("stage") != previous:
                        previous = status["stage"]
                        state.update(stage="waiting_for_pipeline", pipeline_stage=previous, checked_at=now())
                        write_json(status_path, state)
                        print(f"{now()} Waiting for pipeline: {previous}", flush=True)
                    time.sleep(30)
            require(sha256(Path(__file__)) == own_hash, "Audit script changed after launch")
            state.update(stage="auditing", checked_at=now())
            write_json(status_path, state)
            audit = final_audit()
            write_json(RUN / "final_audit.json", audit)
            write_report(audit)
            state.update(state="complete", stage="complete", completed_at=now(),
                         audit_sha256=sha256(RUN / "final_audit.json"))
            write_json(status_path, state)
            print("Independent final audit passed; see SMFA_FINAL_AUDIT.md", flush=True)
        except BaseException as error:
            state.update(state="failed", failed_at=now(), error=str(error))
            write_json(status_path, state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        os.umask(0o077)
        with (RUN / "final_audit.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--watch"],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
        print(f"Launched independent final audit watcher PID {process.pid}")
    else:
        main(args.watch)
