"""Apply the released SMFA formula one layer at a time to bound host memory."""

import json
import sys
from pathlib import Path

import torch
from peft.tuners.lora import Linear as LoraLinear

REPO = Path(__file__).resolve().parents[1] / "S-MLLMUn"
sys.path.insert(0, str(REPO))
from unlearning_methods import SMFA_sculpt as official


def sculpt_delta(retain, forget, k):
    ratio = torch.norm(forget) / (torch.norm(retain) + 1e-12)
    mask = (retain * forget) < 0
    if k >= 0:
        mask &= k * ratio * torch.abs(retain) < torch.abs(forget)
    result = forget.clone()
    result[mask] = 0.0
    return result, int(mask.sum()), mask.numel()


def main(args):
    model = official.load_model_and_adapter(args)
    stats = {"multi": {"masked": 0, "total": 0}, "text": {"masked": 0, "total": 0}}
    layer_count = 0
    with torch.no_grad():
        for name, layer in model.named_modules():
            if not isinstance(layer, LoraLinear):
                continue
            deltas = {}
            for adapter in ("RA", "MFA_multi", "MFA_text"):
                deltas[adapter] = (
                    layer.scaling[adapter]
                    * (layer.lora_B[adapter].weight @ layer.lora_A[adapter].weight)
                ).cpu()
            multi, masked, total = sculpt_delta(deltas["RA"], deltas["MFA_multi"], args.multi_k)
            stats["multi"]["masked"] += masked
            stats["multi"]["total"] += total
            text, masked, total = sculpt_delta(deltas["RA"], deltas["MFA_text"], args.text_k)
            stats["text"]["masked"] += masked
            stats["text"]["total"] += total
            weight = layer.base_layer.weight
            weight.copy_(weight + ((multi + text) / 2).to(weight.device))
            layer_count += 1
            if layer_count % 28 == 0:
                print(f"Sculpted {layer_count} layers: {name}", flush=True)
            del deltas, multi, text
    final_model = model.unload()
    final_model.save_pretrained(args.save_dir)
    for item in stats.values():
        item["ratio"] = item["masked"] / item["total"]
    stats.update(layers=layer_count, multi_k=args.multi_k, text_k=args.text_k)
    (Path(args.save_dir) / "sculpt_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats), flush=True)
    print(f"Model saved to: {args.save_dir}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main(official.parse_arguments())
