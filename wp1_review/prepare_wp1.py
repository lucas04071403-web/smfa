"""Freeze WP1 identity splits and summarize existing image-understanding outputs."""

import argparse
import ast
import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rouge_score.rouge_scorer import RougeScorer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "S-MLLMUn/data/S-MLLMUn-data"
EVAL = ROOT / "S-MLLMUn/eval_result/llava-onevision/split_ratio_5"
OUT = ROOT / "wp1_review"
SEED = 42


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def rows(path):
    frame = pd.read_parquet(path)
    if "ID" not in frame:
        raise ValueError(f"Missing ID column: {path}")
    frame["ID"] = frame["ID"].astype(str)
    if frame["ID"].duplicated().any():
        raise ValueError(f"Duplicate person IDs: {path}")
    return frame


def make_partition():
    forget_path = DATA / "eval_data/forget_5/train-00000-of-00001.parquet"
    retain_path = DATA / "eval_data/retain_5/train-00000-of-00001.parquet"
    fewshot_path = DATA / "unlearning_data/retain_few_shot_5/train-00000-of-00001.parquet"
    forget, retain, fewshot = map(rows, (forget_path, retain_path, fewshot_path))
    forget_ids = sorted(forget["ID"].tolist())
    retain_ids = sorted(retain["ID"].tolist())
    b1_ids = sorted(fewshot["ID"].tolist())
    b1_set = set(b1_ids)
    remaining = sorted(set(retain_ids) - b1_set)
    if len(forget_ids) != 50 or len(retain_ids) != 950 or len(b1_ids) != 50:
        raise ValueError("Unexpected WP1 population size; expected forget=50, retain=950, B1=50")
    if set(forget_ids) & set(retain_ids):
        raise ValueError("Forget and retain identity sets overlap")
    if len(remaining) != 900:
        raise ValueError(f"Expected 900 non-B1 retain identities, found {len(remaining)}")

    rng = random.Random(SEED)
    shuffled_forget = forget_ids.copy()
    rng.shuffle(shuffled_forget)
    f1, f2 = sorted(shuffled_forget[:25]), sorted(shuffled_forget[25:])
    rng.shuffle(remaining)
    dev = sorted(remaining[:100])
    test = sorted(remaining[100:300])
    remainder = remaining[300:]
    x_pool = sorted(remainder[:300])
    h_pool = sorted(remainder[300:])
    partition = {
        "seed": SEED,
        "algorithm": "Python random.Random(seed), sorted IDs before shuffle, sequential slices",
        "forget": {"F1_development": f1, "F2_test": f2},
        "retain": {"B1_few_shot_excluded": b1_ids, "development": dev, "test": test,
                   "X_pool": x_pool, "H_pool": h_pool},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p)
                           for p in (forget_path, retain_path, fewshot_path)},
        "protocol_note": (
            "Split membership used only parquet identity IDs and the fixed seed, not model scores. "
            "A few initial generated records were inspected earlier solely to confirm result schema; "
            "they did not affect split construction."
        ),
    }
    all_groups = [f1, f2, dev, test, x_pool, h_pool, b1_ids]
    flattened = [identity for group in all_groups for identity in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Partition groups overlap")
    if set(f1 + f2) != set(forget_ids) or set(b1_ids + dev + test + x_pool + h_pool) != set(retain_ids):
        raise ValueError("Partition does not cover all evaluation identities")
    atomic_json(OUT / "fixed_person_split.json", partition)
    return partition


def load_partition():
    """Validate the frozen split without changing its membership or manifest."""
    path = OUT / "fixed_person_split.json"
    partition = json.loads(path.read_text())
    if partition["seed"] != SEED:
        raise ValueError("Frozen split has an unexpected seed")
    for name, expected in partition["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Split input changed since freezing: {name}")
    groups = {**partition["forget"], **partition["retain"]}
    sizes = {"F1_development": 25, "F2_test": 25, "B1_few_shot_excluded": 50,
             "development": 100, "test": 200, "X_pool": 300, "H_pool": 300}
    if {name: len(ids) for name, ids in groups.items()} != sizes:
        raise ValueError("Frozen split has unexpected group sizes")
    flat = [pid for ids in groups.values() for pid in ids]
    if len(flat) != len(set(flat)):
        raise ValueError("Frozen split groups overlap")
    for split in ("forget", "retain"):
        expected = set(rows(DATA / f"eval_data/{split}_5/train-00000-of-00001.parquet")["ID"])
        actual = {pid for ids in partition[split].values() for pid in ids}
        if actual != expected:
            raise ValueError(f"Frozen split does not cover the {split} identities")
    few = set(rows(DATA / "unlearning_data/retain_few_shot_5/train-00000-of-00001.parquet")["ID"])
    if few != set(partition["retain"]["B1_few_shot_excluded"]):
        raise ValueError("B1 exclusion does not match the actual RA few-shot identities")
    atomic_json(OUT / "fixed_person_split_validation.json", {
        "state": "passed", "checked_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(path), "counts": sizes,
        "all_groups_disjoint": True, "full_coverage": True, "source_hashes_verified": True,
        "manifest_unchanged": True, "protocol_note": partition["protocol_note"],
    })
    return partition


def refusal_phrases():
    paths = [ROOT / "S-MLLMUn/data_process/IDK.py", ROOT / "S-MLLMUn/data_process/IDK_extend.py"]
    phrases = []
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in {"idk", "idk_extend"}
                for target in node.targets
            ):
                value = ast.literal_eval(node.value)
                phrases.extend(value.values() if isinstance(value, dict) else value)
    return {normalize(phrase) for phrase in phrases}


def normalize(text):
    text = str(text or "").casefold().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())


LOOSE_REFUSAL = re.compile(
    r"\b(?:i\s+(?:do\s+not|don't|cannot|can't|am\s+unable|am\s+not\s+able)\s+"
    r"(?:to\s+)?(?:answer|provide|help|know|tell)|unable\s+to\s+(?:answer|provide|help)|"
    r"not\s+able\s+to\s+(?:answer|provide|help)|i\s+don't\s+know|i\s+am\s+not\s+sure|"
    r"i'm\s+not\s+sure|outside\s+(?:my|the)\s+(?:knowledge|expertise|capabilities)|"
    r"lack\s+(?:the\s+)?(?:information|knowledge|data))\b", re.I,
)


def source_questions(frame):
    expected = []
    for _, row in frame.iterrows():
        person_id = str(row["ID"])
        for kind, column in (("image_understanding", "image understanding"),
                             ("image_memory", "image memory"),
                             ("text_memory", "text memory")):
            for item in row[column]:
                expected.append((person_id, kind, item["Question"], item["Answer"]))
    return expected


def annotate_questions(questions, frame):
    """Recover text-question identities from parquet order and validate images too."""
    expected = source_questions(frame)
    if len(questions) != len(expected):
        raise ValueError("Generation count does not match the source parquet")
    annotated = []
    image_types = [item.get("Image") for _, row in frame.iterrows()
                   for column in ("image understanding", "image memory", "text memory")
                   for item in row[column]]
    for index, (q, (pid, kind, question, answer), image_type) in enumerate(zip(questions, expected, image_types)):
        if (q["question type"], q["question"], q["ground_truth"]) != (kind, question, answer):
            raise ValueError(f"Generation differs from source at question {index}")
        if kind != "text_memory" and (str(q.get("image_id")), q.get("image_type")) != (pid, image_type):
            raise ValueError(f"Generation image identity/type differs at question {index}")
        if q.get("image_id") is not None and str(q["image_id"]) != pid:
            raise ValueError(f"Generation identity differs at question {index}")
        annotated.append((q, pid))
    return annotated


def judge_key(item, scored=False):
    if scored:
        kind, identity = item["Question Type"], item.get("Image_ID")
        fields = (item["Question"], item["Generated Answer"], item["Ground Truth"])
    else:
        kind, identity = item["question type"], item.get("image_id")
        fields = (item["question"], item["generated_answer"], item["ground_truth"])
    return (kind, None if identity is None else str(identity), *fields)


def align_judge_rows(questions, records, field):
    """Join shuffled official score JSON rows; reject ambiguous duplicate questions."""
    buckets = defaultdict(list)
    for record in records:
        buckets[judge_key(record, scored=True)].append(record[field])
    expected_counts = defaultdict(int)
    for q in questions:
        expected_counts[judge_key(q)] += 1
    if set(buckets) != set(expected_counts):
        raise ValueError("Judge questions do not match the generation questions")
    for key, count in expected_counts.items():
        if len(buckets[key]) != count:
            raise ValueError("Judge question multiplicity differs from generation")
        if len(set(buckets[key])) > 1:
            raise ValueError("Ambiguous duplicate judge rows; use the indexed/digested JSONL cache")
    return [buckets[judge_key(q)][0] for q in questions]


def load_judge_scores(folder, split, questions, suffix, field):
    cache = folder / f"{split}_generation_results_{suffix}.jsonl"
    final = folder / f"{split}_generation_results_{suffix}_score.json"
    values = [None] * len(questions)
    used = None
    if cache.exists():
        used = cache
        for line in cache.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # A partially written last line is not a completed score.
            index = record["index"]
            if type(index) is not int or not 0 <= index < len(questions):
                raise ValueError(f"Invalid judge-cache index in {cache}")
            expected_digest = hashlib.sha256(json.dumps(questions[index], sort_keys=True).encode()).hexdigest()
            if record["digest"] != expected_digest:
                continue  # Stale generations are never used.
            result = record["result"]
            if judge_key(result, scored=True) != judge_key(questions[index]):
                raise ValueError(f"Judge tuple mismatch in {cache}, index {index}")
            value = result.get(field)
            if value is not None:
                if type(value) is not int or not 0 <= value <= 10:
                    raise ValueError(f"Invalid {field} in {cache}")
                values[index] = value
    elif final.exists():
        used = final
        data = json.loads(final.read_text())
        records = [item for kind in ("image_understanding", "image_memory", "text_memory")
                   for item in data[kind]]
        values = align_judge_rows(questions, records, field)
        if any(type(value) is not int or not 0 <= value <= 10 for value in values):
            raise ValueError(f"Invalid or missing {field} in {final}")
    return values, {"source": None if used is None else str(used.relative_to(ROOT)),
                    "sha256": None if used is None else sha256(used),
                    "scored_questions": sum(value is not None for value in values),
                    "total_questions": len(questions),
                    "complete": all(value is not None for value in values)}


def summarize(records):
    count = len(records)
    result = {"n": count, "unique_people": len({r["person_id"] for r in records}),
              "exact_refusals": sum(r["exact_refusal"] for r in records),
              "loose_refusals": sum(r["loose_refusal"] for r in records),
              "exact_refusal_rate": sum(r["exact_refusal"] for r in records) / count,
              "loose_refusal_rate": sum(r["loose_refusal"] for r in records) / count,
              "rougeL": sum(r["rougeL"] for r in records) / count,
              "mean_whitespace_words": sum(r["answer_words"] for r in records) / count}
    for metric in ("factuality_score", "meaningful_score"):
        values = [r[metric] for r in records if r[metric] is not None]
        result[metric] = sum(values) / count if len(values) == count else None
        result[metric + "_count"] = len(values)
    return result


def image_group(question):
    kind = question["question type"]
    if kind == "text_memory":
        return "text"
    value = str(question.get("image_type", "")).casefold()
    if value in {"character", "face", "facial", "face_image"}:
        return "face"
    if value in {"ophthalmic", "ophthalmology", "eye", "eye_image"}:
        return "ophthalmology"
    raise ValueError(f"Unknown image_type {value!r}")


def analyze_model(label, exact_phrases):
    partition = json.loads((OUT / "fixed_person_split.json").read_text())
    cohorts = {
        **{identity: group for group in ("F1_development", "F2_test")
           for identity in partition["forget"][group]},
        **{identity: group for group in ("B1_few_shot_excluded", "development", "test", "X_pool", "H_pool")
           for identity in partition["retain"][group]},
    }
    frame_by_split = {
        "forget": rows(DATA / "eval_data/forget_5/train-00000-of-00001.parquet"),
        "retain": rows(DATA / "eval_data/retain_5/train-00000-of-00001.parquet"),
    }
    model_dir = EVAL / label
    model_summary = {}
    provenance = {}
    scorer = RougeScorer(["rougeL"], use_stemmer=True)
    for split, frame in frame_by_split.items():
        result_path = model_dir / f"{split}_generation_results.json"
        questions = json.loads(result_path.read_text())["Generation_Questions"]
        annotated = annotate_questions(questions, frame)
        fact, fact_info = load_judge_scores(model_dir, split, questions, "factuality", "Factuality Score")
        meaning, meaning_info = load_judge_scores(model_dir, split, questions, "meaningful", "Meaningful Score")
        provenance[split] = {"generation_sha256": sha256(result_path),
                             "factuality": fact_info, "meaningful": meaning_info,
                             "question_and_image_identity_coverage_verified": True}
        groups = defaultdict(list)
        for index, (q, person_id) in enumerate(annotated):
            if q["question type"] != "image_understanding":
                continue
            group = image_group(q)
            groups[group].append({
                "person_id": person_id,
                "cohort": cohorts[person_id],
                "question": q["question"],
                "generated_answer": q["generated_answer"],
                "ground_truth": q["ground_truth"],
                "generation_index": index,
                "rougeL": scorer.score(q["ground_truth"], q["generated_answer"])["rougeL"].fmeasure,
                "answer_words": len(q["generated_answer"].split()),
                "factuality_score": fact[index],
                "meaningful_score": meaning[index],
                "exact_refusal": normalize(q["generated_answer"]) in exact_phrases,
                "loose_refusal": (
                    normalize(q["generated_answer"]) in exact_phrases
                    or bool(LOOSE_REFUSAL.search(q["generated_answer"] or ""))
                ),
            })
        split_summary = {}
        for group, records in sorted(groups.items()):
            by_cohort = defaultdict(list)
            for record in records:
                by_cohort[record["cohort"]].append(record)
            split_summary[group] = {**summarize(records), "by_cohort": {
                    cohort: summarize(cohort_records)
                    for cohort, cohort_records in sorted(by_cohort.items())
                }}
        model_summary[split] = split_summary
        out_path = OUT / f"{label}_{split}_image_understanding_split.json"
        atomic_json(out_path, {"model": label, "split": split, "by_image_type": groups})
    return model_summary, provenance


def analyze_existing():
    load_partition()  # Validate the frozen split before reading any model outputs.
    phrases = refusal_phrases()
    labels = ("Original", "SMFA", "Base", "MFA_unsculpted")
    available = [label for label in labels if all(
        (EVAL / label / f"{split}_generation_results.json").exists() for split in ("forget", "retain")
    )]
    outputs = {label: analyze_model(label, phrases) for label in available}
    summaries = {label: output[0] for label, output in outputs.items()}
    atomic_json(OUT / "existing_i_u_split_summary.json", {
        "models": summaries,
        "refusal_definition": {
            "exact": "casefold + punctuation normalization, exact match against IDK.py and IDK_extend.py phrase lists",
            "loose": "regex patterns in prepare_wp1.py; reports broad lexical refusal indicators",
        },
        "score_model": "Existing step-0 qwen-plus scores; no new API calls. Refusals use generated answers only.",
        "score_join": "JSONL index + generation digest + full question tuple; unordered JSON fallback uses full tuple.",
        "split_manifest_sha256": sha256(OUT / "fixed_person_split.json"),
        "provenance": {label: output[1] for label, output in outputs.items()},
        "pending_models": [label for label in labels if label not in available],
    })
    report = ["# 第 0 步：图像理解拆分", "", "仅重新统计已有输出，无生成或评分 API 调用。拒答是词法规则统计。", "",
              "| 模型 | 集合 | 图像 | 题数 | 事实性 | 有意义性 | ROUGE-L | 精确拒答 | 宽松拒答 | 平均词数 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, model in summaries.items():
        for split, groups in model.items():
            for group, stats in groups.items():
                def score_text(name):
                    return "待评分" if stats[name] is None else f"{stats[name]:.3f}"
                report.append(f"| {label} | {split} | {group} | {stats['n']} | {score_text('factuality_score')} | "
                              f"{score_text('meaningful_score')} | {stats['rougeL']:.4f} | "
                              f"{stats['exact_refusal_rate']:.2%} | {stats['loose_refusal_rate']:.2%} | "
                              f"{stats['mean_whitespace_words']:.2f} |")
    report += ["", "Base 和 MFA_unsculpted 的完整结果产生后，重新运行 prepare_wp1.py --analyze-existing 即可补入。", ""]
    (OUT / "step0_image_understanding_report.md").write_text("\n".join(report))
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-existing", action="store_true")
    args = parser.parse_args()
    partition_path = OUT / "fixed_person_split.json"
    if partition_path.exists():
        existing = json.loads(partition_path.read_text())
        if existing["seed"] != SEED:
            raise ValueError("Existing split uses a different seed; refusing to overwrite")
        partition = load_partition()
    else:
        partition = make_partition()
    counts = {name: len(group) for name, group in partition["retain"].items()}
    counts.update({name: len(group) for name, group in partition["forget"].items()})
    print(json.dumps({"partition": str((OUT / 'fixed_person_split.json').relative_to(ROOT)),
                      "counts": counts}, indent=2))
    if args.analyze_existing:
        summaries = analyze_existing()
        print(json.dumps({"analyzed_models": list(summaries),
                          "summary": str(OUT / "existing_i_u_split_summary.json"),
                          "report": str(OUT / "step0_image_understanding_report.md")}, indent=2))


if __name__ == "__main__":
    main()
