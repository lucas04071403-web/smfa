"""Resume official Qwen-Plus judging for both forget and retain generations."""

import argparse
import hashlib
import importlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "S-MLLMUn"
sys.path.insert(0, str(REPO))
KINDS = ("image_understanding", "image_memory", "text_memory")
LABELS = ("Image Understanding", "Image Memory", "Text Memory")


def digest(question):
    return hashlib.sha256(json.dumps(question, sort_keys=True).encode()).hexdigest()


def score_file(module, path, field, suffix, workers):
    questions = json.loads(path.read_text())["Generation_Questions"]
    cache = path.with_name(path.stem + f"_{suffix}.jsonl")
    records = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            index = row["index"]
            if index < len(questions) and row["digest"] == digest(questions[index]):
                records[index] = row["result"]

    def task(index):
        buckets = {kind: [] for kind in KINDS}
        scores = {kind: [] for kind in KINDS}
        module.process_single_question(questions[index], buckets, scores)
        return index, buckets[questions[index]["question type"]][0]

    pending = [i for i in range(len(questions)) if records.get(i, {}).get(field) is None]
    print(f"{path.name} {suffix}: {len(questions) - len(pending)}/{len(questions)} cached", flush=True)
    with cache.open("a") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(task, index) for index in pending]
        for future in as_completed(futures):
            index, result = future.result()
            records[index] = result
            handle.write(json.dumps({"index": index, "digest": digest(questions[index]), "result": result}) + "\n")
            handle.flush()
            if len(records) % 100 == 0:
                print(f"{path.name} {suffix}: {len(records)}/{len(questions)} recorded", flush=True)
    missing = [i for i in range(len(questions)) if records.get(i, {}).get(field) is None]
    if missing:
        raise RuntimeError(f"{len(missing)} questions have no {field}; resume to retry them")
    output = {kind: [] for kind in KINDS}
    for index in range(len(questions)):
        output[questions[index]["question type"]].append(records[index])
    output["Score"] = {f"Average {label} Score": sum(q[field] for q in output[kind]) / len(output[kind])
                       for kind, label in zip(KINDS, LABELS)}
    destination = path.with_name(path.stem + f"_{suffix}_score.json")
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=4) + "\n")
    temporary.replace(destination)
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for module_name, field, suffix in (("eval_fact", "Factuality Score", "factuality"),
                                       ("eval_meaningful", "Meaningful Score", "meaningful")):
        module = importlib.import_module(module_name)
        if module.client is None:
            raise RuntimeError("Activate the smllmun Conda environment to load the API credential")
        module.client = module.client.with_options(timeout=120, max_retries=2)
        for split in ("forget", "retain"):
            score_file(module, args.input_folder / f"{split}_generation_results.json", field, suffix, args.workers)
