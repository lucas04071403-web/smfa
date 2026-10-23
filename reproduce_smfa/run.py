"""Sequential, restartable reproduction of the README's default SMFA experiment."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "S-MLLMUn"
TOOLS = ROOT / "reproduce_smfa"
RUN = ROOT / "smfa_run"
MODEL_ID = "model/base_model/llava-onevision-qwen2-7b-si-hf"
ORIGINAL = "model/original_model/llava-onevision-qwen2-7b-original"
MODEL = "model/unlearned_model/llava-onevision/split_ratio_5/SMFA"
EVAL_ROOT = "eval_result/llava-onevision/split_ratio_5"
KINDS = ("image_understanding", "image_memory", "text_memory")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def command(script, *arguments):
    return [sys.executable, "-u", str(TOOLS / script), *arguments]


def stages():
    common = ["--model_id", MODEL_ID, "--original_dir", ORIGINAL]
    result = [("train", command("train.py", *common, "--data_split_dir", "data/S-MLLMUn-data/unlearning_data",
                 "--forget_split_ratio", "5", "--save_dir", MODEL, "--batch_size", "2", "--lr", "1e-4", "--num_epochs", "5"))]
    result.append(("sculpt", command("sculpt.py", *common, "--save_dir", f"{MODEL}/sculpted",
                   "--MFA_multi", f"{MODEL}/epoch_3/MFA_multi", "--MFA_text", f"{MODEL}/epoch_3/MFA_text",
                   "--RA", f"{MODEL}/epoch_3/RA", "--multi_k", "5", "--text_k", "10")))
    for label, checkpoint in (("SMFA", f"{MODEL}/sculpted"), ("Original", ORIGINAL)):
        folder = f"{EVAL_ROOT}/{label}"
        result.append((f"evaluate_{label}", command("evaluate.py", "--model_id", MODEL_ID,
                       "--cache_path", checkpoint, "--data_split_folder", "data/S-MLLMUn-data/eval_data",
                       "--output_file", label, "--output_folder", folder, "--forget_ratio", "5")))
        result.append((f"score_{label}", command("score.py", "--input_folder", folder, "--workers", "4")))
    return result


def check_stage(name):
    if name == "train":
        from safetensors import safe_open
        for epoch in range(1, 6):
            for adapter in ("MFA_multi", "MFA_text", "RA"):
                path = REPO / MODEL / f"epoch_{epoch}" / adapter / "adapter_model.safetensors"
                with safe_open(path, framework="pt", device="cpu") as tensors:
                    import torch
                    keys = list(tensors.keys())
                    if not keys or not all(bool(torch.isfinite(tensors.get_tensor(k)).all()) for k in keys):
                        raise RuntimeError(f"Invalid adapter checkpoint: {path}")
                    b_keys = [k for k in keys if "lora_B" in k]
                    if not b_keys or not any(bool(tensors.get_tensor(k).count_nonzero()) for k in b_keys):
                        raise RuntimeError(f"Adapter has no learned delta: {path}")
        return
    if name == "sculpt":
        from safetensors import safe_open
        folder = REPO / MODEL / "sculpted"
        index = json.loads((folder / "model.safetensors.index.json").read_text())
        found = set()
        for shard in set(index["weight_map"].values()):
            with safe_open(folder / shard, framework="pt", device="cpu") as tensors:
                found.update(tensors.keys())
        if found != set(index["weight_map"]):
            raise RuntimeError("Sculpted checkpoint has missing tensor shards")
        stats = json.loads((folder / "sculpt_stats.json").read_text())
        if not stats["layers"]:
            raise RuntimeError("No layers sculpted")
        return
    label = name.split("_", 1)[1]
    folder = REPO / EVAL_ROOT / label
    import pandas as pd
    sys.path.insert(0, str(TOOLS))
    from evaluate import verify_questions
    for split in ("forget", "retain"):
        source = REPO / "data/S-MLLMUn-data/eval_data" / f"{split}_5/train-00000-of-00001.parquet"
        questions = json.loads((folder / f"{split}_generation_results.json").read_text())["Generation_Questions"]
        verify_questions(questions, pd.read_parquet(source))
        if name.startswith("score_"):
            from collections import Counter
            counts = Counter(q["question type"] for q in questions)
            for suffix, field in (("factuality", "Factuality Score"), ("meaningful", "Meaningful Score")):
                scores = json.loads((folder / f"{split}_generation_results_{suffix}_score.json").read_text())
                for kind in KINDS:
                    if len(scores[kind]) != counts[kind]:
                        raise RuntimeError(f"Incomplete {suffix} coverage for {label}/{split}/{kind}")
                    for source_question, score in zip((q for q in questions if q["question type"] == kind), scores[kind]):
                        if source_question["question"] != score["Question"] or source_question["generated_answer"] != score["Generated Answer"]:
                            raise RuntimeError("Judge score refers to different generation")
                        if not isinstance(score[field], int) or not 0 <= score[field] <= 10:
                            raise RuntimeError(f"Invalid or absent {field}")
    if name.startswith("evaluate_"):
        final = folder / f"{label}_final_evaluation_results.json"
        if not final.exists():
            raise RuntimeError("Final ROUGE results are missing")


def report():
    summary = {}
    lines = ["# SMFA README Reproduction", "", "Configuration: LLaVA-OneVision, forget ratio 5%, batch size 2, learning rate 1e-4, 5 full epochs.",
             "Sculpting follows the released template: epoch 3, multi_k=5, text_k=10. Seed: 42.",
             "Single physical GPU 1 with gradient checkpointing. Official optimizer, scheduler, data collator and prompts are retained.",
             "", "| Model | Split | Questions | ROUGE-L IU | ROUGE-L IM | ROUGE-L TM | Fact IU | Fact IM | Fact TM | Meaningful IU | Meaningful IM | Meaningful TM |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label in ("Original", "SMFA"):
        folder = REPO / EVAL_ROOT / label
        rouge = json.loads((folder / f"{label}_final_evaluation_results.json").read_text())
        summary[label] = {"rouge": rouge}
        for split, key in (("forget", "Forget Set Results"), ("retain", "Retain Set (shared dataset) Results")):
            values = list(rouge[key]["generation"].values())
            n = len(json.loads((folder / f"{split}_generation_results.json").read_text())["Generation_Questions"])
            summary[label][split] = {"questions": n}
            for suffix in ("factuality", "meaningful"):
                scores = json.loads((folder / f"{split}_generation_results_{suffix}_score.json").read_text())["Score"]
                summary[label][split][suffix] = scores
                values.extend(scores.values())
            lines.append(f"| {label} | {split} | {n} | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    lines += ["", "ROUGE-L is on a 0-1 scale; API judge scores are on a 0-10 scale.",
              "This reproduces the README default SMFA run and evaluates the supplied Original checkpoint for comparison.",
              "It does not establish reproduction of every baseline, ratio, backbone, or ablation in the paper.",
              "The paper's stated k=5 differs from the released text_k=10 template used here."]
    write_json(RUN / "results.json", summary)
    (ROOT / "SMFA_REPRODUCTION.md").write_text("\n".join(lines) + "\n")


def main():
    RUN.mkdir(exist_ok=True)
    lock = (RUN / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.chdir(REPO)
    plan = stages()
    manifest = {"created_at": now(), "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "conda_prefix": sys.prefix, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "seed": 42,
                "commands": dict(plan), "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                                           for path in sorted(TOOLS.glob("*.py"))}}
    manifest_path = RUN / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["commands"] != manifest["commands"] or previous["source_sha256"] != manifest["source_sha256"]:
            raise RuntimeError("Configuration or runner changed; inspect the existing run before reusing its outputs")
    else:
        write_json(manifest_path, manifest)
    status_path = RUN / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"completed_stages": []}
    status.update(pid=os.getpid(), started_at=now(), state="running")
    write_json(status_path, status)
    try:
        for name, cmd in plan:
            if name in status["completed_stages"]:
                check_stage(name)
                continue
            status.update(stage=name, stage_started_at=now(), state="running")
            write_json(status_path, status)
            print(f"{now()} Starting {name}", flush=True)
            with (RUN / f"{name}.log").open("a") as log:
                child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                status["child_pid"] = child.pid
                write_json(status_path, status)
                code = child.wait()
            if code:
                raise RuntimeError(f"{name} exited with code {code}; see {RUN / (name + '.log')}")
            check_stage(name)
            status["completed_stages"].append(name)
            status.pop("child_pid", None)
            write_json(status_path, status)
            print(f"{now()} Verified {name}", flush=True)
        report()
        status.update(state="complete", completed_at=now(), stage="complete")
        write_json(status_path, status)
    except BaseException as error:
        status.update(state="failed", error=str(error), failed_at=now())
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        RUN.mkdir(exist_ok=True)
        with (RUN / "pipeline.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        print(f"Launched reproduction pipeline PID {process.pid}; status: {RUN / 'status.json'}")
    else:
        main()
