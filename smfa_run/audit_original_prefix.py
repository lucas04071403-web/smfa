"""Audit closed Original retain prefixes without GPU or API requests."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import audit_final as auditor

ROOT = auditor.ROOT
RUN = auditor.RUN


def cache_snapshot(path):
    data = path.read_bytes()
    auditor.require(data.endswith(b"\n"), f"Incomplete cache record: {path}")
    lines = data.splitlines(keepends=True)
    records = {}
    for line in lines:
        record = json.loads(line)
        index = record["index"]
        auditor.require(type(index) is int and index >= 0, f"Invalid cache index: {path}")
        auditor.require(index not in records, f"Duplicate cache index: {path}/{index}")
        records[index] = record
    return lines, records


def closed_prefix(profiles):
    name = f"early_original_retain_{profiles}"
    state = auditor.read_json(RUN / f"{name}_status.json")
    auditor.require(state["state"] == "complete", f"Prefix is not complete: {name}")
    auditor.require(set(state["completed_metrics"]) == {"factuality", "meaningful"},
                    f"Prefix lacks both metrics: {name}")
    auditor.require((state["profiles"], state["questions"]) == (profiles, profiles * 19),
                    f"Prefix coverage differs: {name}")
    folder = RUN / name
    auditor.require(auditor.sha256(folder / "retain_generation_results.json") == state["input_sha256"],
                    f"Judged input changed: {name}")
    return folder, state


def audit(profiles, previous):
    import pandas as pd

    manifest = auditor.read_json(RUN / "manifest.json")
    auditor.require(sys.prefix == manifest["conda_prefix"], "Activate the recorded Conda environment")
    for relative, expected in manifest["source_sha256"].items():
        auditor.require(auditor.sha256(ROOT / relative) == expected, f"Experiment source changed: {relative}")
    folder, state = closed_prefix(profiles)
    dataset = auditor.REPO / "data/S-MLLMUn-data/eval_data/retain_5/train-00000-of-00001.parquet"
    rows = pd.read_parquet(dataset)
    auditor.require(0 < profiles < len(rows), "This auditor requires a partial retain prefix")
    measured = auditor.audit_split(folder, "retain", rows.iloc[:profiles])
    controller = auditor.read_json(RUN / "parallel_original_status.json")
    published = Path(controller["staging"])
    if not published.exists():
        published = Path(controller["destination"])
    questions = auditor.read_json(folder / "retain_generation_results.json")["Generation_Questions"]
    original_questions = []
    for index in range(profiles):
        source = published / ".profiles/retain" / f"{index:04d}" / "retain_generation_results.json"
        original_questions.extend(auditor.read_json(source)["Generation_Questions"])
    auditor.require(original_questions == questions, "Judged answers differ from the original generator")
    prior_folder = None
    prior_report = None
    if previous is not None:
        auditor.require(0 < previous < profiles, "The previous prefix must be smaller")
        prior_folder, _ = closed_prefix(previous)
        prior_report = auditor.read_json(RUN / f"partial_original_retain_{previous}.json")
        auditor.require((prior_report["profiles"], prior_report["questions"]) == (previous, previous * 19),
                        "Previous audit coverage differs")
    publication = {}
    for metric in ("factuality", "meaningful"):
        name = f"retain_generation_results_{metric}.jsonl"
        _, local = cache_snapshot(folder / name)
        shared_lines, shared = cache_snapshot(published / name)
        count = len(questions)
        auditor.require(set(local) == set(range(count)), f"Incomplete local {metric} cache")
        shared_prefix = {index: record for index, record in shared.items() if index < count}
        auditor.require(shared_prefix == local, f"Published {metric} cache differs from validated scores")
        prefix_lines = shared_lines[:count]
        auditor.require({json.loads(line)["index"] for line in prefix_lines} == set(range(count)),
                        f"Published {metric} records are not a complete leading prefix")
        entry = {"records": count,
                 "sha256": hashlib.sha256(b"".join(prefix_lines)).hexdigest(),
                 "snapshot_total_records": len(shared)}
        if prior_folder is not None:
            _, prior = cache_snapshot(prior_folder / name)
            auditor.require(len(prior) == previous * 19, f"Incomplete prior {metric} cache")
            auditor.require(all(local.get(index) == record for index, record in prior.items()),
                            f"A reused {metric} score changed")
            prior_hash = hashlib.sha256(b"".join(shared_lines[:previous * 19])).hexdigest()
            auditor.require(prior_hash == prior_report["published_caches"][metric]["sha256"],
                            f"Previously published {metric} bytes changed")
            entry.update(previous_records_unchanged=len(prior),
                         previous_published_bytes_unchanged=True)
        publication[metric] = entry
    auditor.require(not (published / "retain_generation_results_factuality_score.json").exists()
                    or (published / "retain_generation_results.json").exists(),
                    "Partial factuality summary was published as a full result")
    auditor.require(not (published / "retain_generation_results_meaningful_score.json").exists()
                    or (published / "retain_generation_results.json").exists(),
                    "Partial meaningfulness summary was published as a full result")
    for name, expected in measured["sha256"].items():
        auditor.require(auditor.sha256(folder / name) == expected, f"Audited input changed during inspection: {name}")
    result = {"state": "passed", "checked_at": auditor.now(), "completed_at": state["completed_at"],
              "scope": "Original retain prefix only; full retain and SMFA results remain pending",
              "profiles": profiles, "total_retain_profiles": len(rows), "questions": len(questions),
              "judge_scores_verified": len(questions) * 2, "published_caches": publication,
              "dataset_sha256": auditor.sha256(dataset), "audit": measured,
              "auditor_sha256": auditor.sha256(Path(__file__)),
              "split_auditor_sha256": auditor.sha256(Path(auditor.__file__)),
              "source_sha256": manifest["source_sha256"],
              "question_weighted_factuality": measured["question_weighted_factuality"],
              "question_weighted_meaningful": measured["question_weighted_meaningful"],
              "cache_hash_note": "Hashes cover only this prefix's leading published records; later appends are allowed."}
    for kind in auditor.KINDS:
        result[kind] = {"questions": measured["category_counts"][kind],
                        "rougeL": measured["rougeL"][kind],
                        "factuality_mean": measured["factuality"][kind],
                        "meaningful_mean": measured["meaningful"][kind]}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=int, required=True)
    parser.add_argument("--previous", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        auditor.require(not args.output.exists(), "Refusing to replace an existing audit record")
    result = audit(args.profiles, args.previous)
    if args.output is not None:
        auditor.write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
