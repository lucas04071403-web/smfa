"""Offline WP1 review evidence from existing generations, caches and source data.

Runs on CPU, makes no API calls, and does not create a WP1 dev/test split.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import runpy
import unicodedata

import numpy as np
import pandas as pd
from rouge_score.rouge_scorer import RougeScorer

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "S-MLLMUn"
DATA = REPO / "data/S-MLLMUn-data"
EVAL = REPO / "eval_result/llava-onevision/split_ratio_5"
KINDS = {"image_understanding": "image understanding", "image_memory": "image memory",
         "text_memory": "text memory"}


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return " ".join(re.findall(r"\w+", text))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    phrases = runpy.run_path(str(REPO / "data_process/IDK_extend.py"))["idk_extend"].values()
    refusals = {normalize(s) for s in phrases}
    hashes = {}
    profiles = {}
    score = RougeScorer(["rougeL"], use_stemmer=True)
    evidence = {"created_at": datetime.now(timezone.utc).isoformat(), "scope":
                "Descriptive reanalysis of existing full results; no WP1 model was run or selected",
                "api_calls": 0, "new_generation_calls": 0, "groups": {}, "paired_differences": {},
                "refusal_rule": "Exact normalized full-answer match against MFA IDK_extend phrases only; not semantic refusal detection",
                "bootstrap": {"unit": "profile", "paired": True, "replicates": 2000,
                              "seed": 20261022, "interval": "percentile 95%; descriptive, unadjusted",
                              "uncertainty_excluded": "training-seed and API-judge variability"}}
    for split in ("forget", "retain"):
        path = DATA / f"eval_data/{split}_5/train-00000-of-00001.parquet"
        hashes[str(path.relative_to(ROOT))] = digest(path)
        rows = pd.read_parquet(path)
        profiles[split] = set(rows["ID"].astype(str))
        expected = [(str(row["ID"]), kind, item) for _, row in rows.iterrows()
                    for kind, col in KINDS.items() for item in row[col]]
        per_model = {}
        for model in ("Original", "SMFA"):
            folder = EVAL / model
            source = folder / f"{split}_generation_results.json"
            questions = json.loads(source.read_text())["Generation_Questions"]
            assert len(questions) == len(expected)
            cache = folder / f"{split}_generation_results_factuality.jsonl"
            records = {}
            for line in cache.read_text().splitlines():
                record = json.loads(line)
                index = record["index"]
                assert index not in records
                assert record["digest"] == hashlib.sha256(
                    json.dumps(questions[index], sort_keys=True).encode()).hexdigest()
                value = record["result"]["Factuality Score"]
                assert type(value) is int and 0 <= value <= 10
                records[index] = value
            assert set(records) == set(range(len(questions)))
            for path in (source, cache):
                hashes[str(path.relative_to(ROOT))] = digest(path)
            per_group = defaultdict(list)
            per_profile = defaultdict(lambda: defaultdict(list))
            for index, (q, (pid, kind, item)) in enumerate(zip(questions, expected)):
                assert (q["question type"], q["question"], q["ground_truth"]) == (
                    kind, item["Question"], item["Answer"])
                image = q.get("image_type", "none")
                if kind != "text_memory":
                    assert str(q["image_id"]) == pid and image == item["Image"]
                rouge = score.score(q["ground_truth"], q["generated_answer"])["rougeL"].fmeasure
                vector = [records[index], rouge, int(normalize(q["generated_answer"]) in refusals),
                          len(q["generated_answer"].split())]
                for group in dict.fromkeys((f"{kind}/{image}", f"{kind}/all")):
                    per_group[group].append(vector)
                    per_profile[group][pid].append(vector[:2])
            evidence["groups"][f"{model}/{split}"] = {
                key: dict(zip(("factuality", "rougeL", "exact_training_refusal_rate", "mean_whitespace_words"),
                              np.asarray(values).mean(axis=0).tolist()), questions=len(values),
                          profiles=len(per_profile[key])) for key, values in per_group.items()}
            per_model[model] = per_profile
        for key in per_model["Original"]:
            pids = sorted(per_model["Original"][key])
            assert set(pids) == set(per_model["SMFA"][key])
            delta = np.array([np.mean(per_model["SMFA"][key][pid], axis=0)
                              - np.mean(per_model["Original"][key][pid], axis=0) for pid in pids])
            rng = np.random.default_rng(20261022)
            samples = delta[rng.integers(len(pids), size=(2000, len(pids)))].mean(axis=1)
            evidence["paired_differences"][f"{split}/{key}"] = {
                metric: {"mean": float(delta[:, i].mean()),
                         "ci95": np.quantile(samples[:, i], [0.025, 0.975]).tolist()}
                for i, metric in enumerate(("factuality", "rougeL"))}
        print(f"Verified and regrouped {split}: {len(expected)} questions per model", flush=True)
    few_path = DATA / "unlearning_data/retain_few_shot_5/train-00000-of-00001.parquet"
    few = pd.read_parquet(few_path)
    few_ids = set(few["ID"].astype(str))
    hashes[str(few_path.relative_to(ROOT))] = digest(few_path)
    evidence["profile_inventory"] = {"forget": len(profiles["forget"]), "retain": len(profiles["retain"]),
                                     "few_shot_retain": len(few_ids),
                                     "few_shot_in_retain": len(few_ids & profiles["retain"]),
                                     "few_shot_in_forget": len(few_ids & profiles["forget"]),
                                     "retain_after_few_shot_exclusion": len(profiles["retain"] - few_ids)}
    evidence["few_shot_qa_types"] = dict(Counter(q["Type"] for row in few["metadata"] for q in json.loads(row)))
    evidence["input_sha256"] = hashes
    destination = ROOT / "wp1_review/existing_results_evidence.json"
    destination.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for split in ("forget", "retain"):
        for group in ("image_understanding/character", "image_understanding/ophthalmic", "image_memory/all", "text_memory/all"):
            old = evidence["groups"][f"Original/{split}"][group]
            new = evidence["groups"][f"SMFA/{split}"][group]
            ci = evidence["paired_differences"][f"{split}/{group}"]["factuality"]["ci95"]
            print(split, group, "n=", old["questions"], "Fact=", round(old["factuality"], 6),
                  "->", round(new["factuality"], 6), "delta CI=", [round(x, 6) for x in ci],
                  "ROUGE=", round(old["rougeL"], 6), "->", round(new["rougeL"], 6))
    print(evidence["profile_inventory"])
    print(evidence["few_shot_qa_types"])
    print(destination)


if __name__ == "__main__":
    main()
