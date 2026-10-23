"""Audit local SMFA assets and optionally run a temporary GPU smoke check."""

import argparse
import gc
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "S-MLLMUn"
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main(smoke, gradient_checkpointing):
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    from packaging.requirements import Requirement
    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "prefix": sys.prefix,
        "dependencies": {},
        "datasets": [],
        "models": [],
        "entrypoints": [],
        "smoke": [],
        "gradient_checkpointing": gradient_checkpointing,
        "limitations": [],
    }

    def save():
        (ROOT / "SMFA_READINESS.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        for line in (REPO / "requirements.txt").read_text().splitlines():
            if not line or line.startswith(("#", "--")):
                continue
            req = Requirement(line)
            version = metadata.version(req.name)
            assert req.specifier.contains(version), (req.name, version, str(req.specifier))
            report["dependencies"][req.name] = version
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
        print("PASS: all official dependency pins and pip check", flush=True)

        expected = [REPO / "data/S-MLLMUn-data/ft_data/ft_data.parquet"]
        for ratio in (5, 10, 15):
            for group, splits in (
                ("unlearning_data", ("forget", "retain_few_shot")),
                ("eval_data", ("forget", "retain")),
            ):
                for split in splits:
                    expected.append(REPO / f"data/S-MLLMUn-data/{group}/{split}_{ratio}/train-00000-of-00001.parquet")
        for path in expected:
            table = pq.read_table(path)
            assert table.num_rows > 0
            row = table.slice(0, 1).to_pylist()[0]
            images = {}
            for key in ("character_image", "ophthalmic_image"):
                image = Image.open(io.BytesIO(row[key]["bytes"]))
                image.load()
                images[key] = image.size
            if "metadata" in row:
                assert isinstance(json.loads(row["metadata"]), list)
            else:
                assert {"image memory", "image understanding", "text memory"} <= row.keys()
            record = {"path": str(path.relative_to(REPO)), "rows": table.num_rows, "columns": table.column_names, "sample_images": images}
            report["datasets"].append(record)
            print("PASS data:", record["path"], record["rows"], flush=True)

        manifest = json.loads((REPO / "model/DOWNLOAD_MANIFEST.json").read_text())
        for item in manifest["models"]:
            path = Path(item["directory"])
            for file in item["files"]:
                assert (path / file["path"]).stat().st_size == file["bytes"], file["path"]
            config = AutoConfig.from_pretrained(path, local_files_only=True)
            processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            index = json.loads((path / "model.safetensors.index.json").read_text())
            found = set()
            for shard in set(index["weight_map"].values()):
                with safe_open(path / shard, framework="pt", device="cpu") as handle:
                    keys = set(handle.keys())
                    for key in keys:
                        assert index["weight_map"].get(key) == shard, key
                    found.update(keys)
            assert found == set(index["weight_map"])
            report["models"].append({"repo_id": item["repo_id"], "directory": str(path), "files": len(item["files"]), "tensors": len(found), "model_type": config.model_type, "processor": type(processor).__name__})
            print("PASS model files/config/processor/tensor index:", item["repo_id"], flush=True)

        for path in [REPO / "finetune.py", REPO / "eval.py", REPO / "eval_fact.py", REPO / "eval_meaningful.py", *sorted((REPO / "unlearning_methods").glob("*.py"))]:
            runpy.run_path(str(path), run_name="readiness_import")
            report["entrypoints"].append(str(path.relative_to(REPO)))
        print("PASS: all training/evaluation entrypoint imports", flush=True)

        assert torch.cuda.is_available()
        x = torch.randn(128, 128, device="cuda", requires_grad=True)
        (x @ x).square().mean().backward()
        torch.cuda.synchronize()
        report["cuda"] = {"torch": torch.__version__, "runtime": torch.version.cuda, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": torch.cuda.get_device_name(0), "bf16": torch.cuda.is_bf16_supported()}
        del x
        torch.cuda.empty_cache()
        print("PASS: CUDA matrix multiplication and backward", flush=True)

        if smoke:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import LlavaOnevisionForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
            from data_process.SMFA_data_preprocess import SMFA_MFA_Dataset, train_collate_fn

            data_root = REPO / "data/S-MLLMUn-data/unlearning_data"
            forget = pd.read_parquet(data_root / "forget_5/train-00000-of-00001.parquet").iloc[:1]
            retain = pd.read_parquet(data_root / "retain_few_shot_5/train-00000-of-00001.parquet").iloc[:1]
            dataset = SMFA_MFA_Dataset(forget, retain, target_size=(384, 384), multimodal=True)
            for family, base, original, cls in (
                ("llava-onevision", "llava-onevision-qwen2-7b-si-hf", "llava-onevision-qwen2-7b-original", LlavaOnevisionForConditionalGeneration),
                ("qwen2.5", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B-original", Qwen2_5_VLForConditionalGeneration),
            ):
                print("Loading original model for GPU smoke:", family, flush=True)
                processor = AutoProcessor.from_pretrained(REPO / "model/base_model" / base, local_files_only=True)
                processor.tokenizer.padding_side = "right"
                model = cls.from_pretrained(REPO / "model/original_model" / original, torch_dtype=torch.float16, device_map={"": 0}, local_files_only=True, low_cpu_mem_usage=True, attn_implementation="sdpa")
                batch = train_collate_fn([dataset[0]], processor, SimpleNamespace(model_id=family, max_length=384))
                batch = {k: v.to("cuda") for k, v in batch.items()}
                model.eval()
                with torch.inference_mode():
                    generated = model.generate(**{k: v for k, v in batch.items() if k != "labels"}, max_new_tokens=2, do_sample=False)
                generated_tokens = int(generated.shape[-1] - batch["input_ids"].shape[-1])
                del generated
                model = prepare_model_for_kbit_training(model)
                config = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], init_lora_weights="gaussian")
                model = get_peft_model(model, config, adapter_name="MFA_multi")
                if gradient_checkpointing:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    model.config.use_cache = False
                model.train()
                output = model(**batch)
                assert torch.isfinite(output.loss)
                loss = float(output.loss.detach())
                output.loss.backward()
                grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
                report["smoke"].append({"family": family, "generated_tokens": generated_tokens, "input_tokens": int(batch["input_ids"].shape[-1]), "multimodal_lora_loss": loss, "gradient_tensors": len(grads), "peak_gpu_bytes": torch.cuda.max_memory_allocated()})
                print("PASS GPU generation + official multimodal LoRA backward:", report["smoke"][-1], flush=True)
                del model, output, batch, grads
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                save()

        if not (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            report["limitations"].append("Qwen-Plus API key is absent; Fact Score and Meaningful Score cannot run yet.")
        report["limitations"].append("Official shell templates list up to six GPU IDs; this host has two GPUs. Select available devices before running.")
        report["limitations"].append("The paper states k=5; the official SMFA shell template uses multi_k=5 and text_k=10. Decide the reproduction configuration before comparing metrics.")
        report["limitations"].append("Full experiments and final metrics have not been run; smoke checks do not establish paper-result reproducibility.")
        if gradient_checkpointing:
            report["limitations"].append("The initial single-GPU LLaVA LoRA check ran out of memory with the official preparation. This retry enables gradient checkpointing; official training scripts have not been changed.")
        report["state"] = "local_checks_passed"
    except Exception:
        report["state"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    main(args.smoke, args.gradient_checkpointing)
