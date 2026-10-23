"""Independently verify closed SMFA retain prefixes and cache preservation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import audit_final as auditor

ROOT = auditor.ROOT
RUN = auditor.RUN
CANONICAL = auditor.REPO / "eval_result/llava-onevision/split_ratio_5/SMFA"


def records(path):
    data = path.read_bytes()
    auditor.require(data.endswith(b"\n"), f"Incomplete cache record: {path}")
    output = {}
    for line in data.splitlines(keepends=True):
        item = json.loads(line)
        index = item["index"]
        auditor.require(type(index) is int and index >= 0 and index not in output,
                        f"Invalid or duplicate cache index: {path}")
        output[index] = line
    return output


def audit(profiles, previous=None):
    import pandas as pd

    manifest = auditor.read_json(RUN / "manifest.json")
    auditor.require(sys.prefix == manifest["conda_prefix"], "Use the recorded Conda environment")
    for name, expected in manifest["source_sha256"].items():
        auditor.require(auditor.sha256(ROOT / name) == expected, f"Frozen source changed: {name}")
    name = f"early_smfa_retain_{profiles}"
    folder = RUN / name
    status_path = RUN / f"{name}_status.json"
    state = auditor.read_json(status_path)
    auditor.require(state["state"] == "complete" and state.get("published") is True,
                    "Both metrics must be complete and published")
    auditor.require(set(state["completed_metrics"]) == {"factuality", "meaningful"},
                    "One judge metric is missing")
    auditor.require((state["profiles"], state["questions"]) == (profiles, 19 * profiles),
                    "Closed prefix coverage differs")
    auditor.require(auditor.sha256(folder / "retain_generation_results.json") == state["input_sha256"],
                    "Judged generation changed")
    auditor.require(auditor.sha256(folder / "prefix_audit.json") == state["audit_sha256"],
                    "Queue audit changed")
    dataset = auditor.REPO / "data/S-MLLMUn-data/eval_data/retain_5/train-00000-of-00001.parquet"
    rows = pd.read_parquet(dataset)
    auditor.require(0 < profiles <= len(rows), "Invalid retain prefix")
    measured = auditor.audit_split(folder, "retain", rows.iloc[:profiles])
    questions = auditor.read_json(folder / "retain_generation_results.json")["Generation_Questions"]
    source_questions = []
    profile_hashes = {}
    for index in range(profiles):
        path = CANONICAL / ".profiles/retain" / f"{index:04d}" / "retain_generation_results.json"
        source_questions.extend(auditor.read_json(path)["Generation_Questions"])
        profile_hashes[str(path.relative_to(CANONICAL))] = auditor.sha256(path)
    auditor.require(source_questions == questions, "Source profiles differ from judged answers")

    previous_folder = None
    previous_report_path = None
    if previous is not None:
        auditor.require(0 < previous < profiles, "Invalid prior prefix")
        previous_folder = RUN / f"early_smfa_retain_{previous}"
        previous_report_path = RUN / f"verification_smfa_retain_{previous}.json"
        previous_report = auditor.read_json(previous_report_path)
        auditor.require(previous_report["state"] == "passed" and previous_report["profiles"] == previous,
                        "Prior independent audit is unavailable")
        for filename, expected in previous_report["split"]["sha256"].items():
            auditor.require(auditor.sha256(previous_folder / filename) == expected,
                            f"Prior audited artifact changed: {filename}")

    published = {}
    prior_preserved = {}
    for metric in ("factuality", "meaningful"):
        cache_name = f"retain_generation_results_{metric}.jsonl"
        local = records(folder / cache_name)
        shared_path = CANONICAL / cache_name
        shared = records(shared_path)
        auditor.require(set(local) == set(range(len(questions))), "Local score coverage differs")
        auditor.require(all(shared.get(index) == line for index, line in local.items()),
                        f"Published {metric} record bytes differ")
        published[cache_name] = {"sha256_at_snapshot": auditor.sha256(shared_path),
                                 "snapshot_records": len(shared), "verified_prefix_records": len(local),
                                 "verified_record_bytes_sha256": hashlib.sha256(
                                     b"".join(local[index] for index in sorted(local))).hexdigest()}
        if previous_folder is not None:
            prior = records(previous_folder / cache_name)
            auditor.require(len(prior) == previous * 19 and all(local.get(i) == line for i, line in prior.items()),
                            f"Reused {metric} record bytes changed")
            prior_preserved[metric] = len(prior)

    factuality_report_path = RUN / f"verification_smfa_retain_{profiles}_factuality.json"
    factuality_report_hash = None
    if factuality_report_path.exists():
        fact = auditor.read_json(factuality_report_path)
        auditor.require(fact["state"] == "passed" and fact["profiles"] == profiles,
                        "Earlier factuality verification has a different scope")
        for relative, expected in fact["sha256"].items():
            auditor.require(auditor.sha256(ROOT / relative) == expected,
                            f"Earlier factuality artifact changed: {relative}")
        factuality_report_hash = auditor.sha256(factuality_report_path)
    for filename, expected in measured["sha256"].items():
        auditor.require(auditor.sha256(folder / filename) == expected,
                        f"An audited prefix artifact changed: {filename}")
    return {"state": "passed", "checked_at": auditor.now(), "profiles": profiles,
            "scope": "SMFA retain prefix only; full retain and final experiment remain pending",
            "generated_answers_verified": len(questions), "judge_scores_verified": 2 * len(questions),
            "split": measured, "profile_sha256": profile_hashes, "data_sha256": auditor.sha256(dataset),
            "published_caches": published, "published_record_bytes_verified_by_index": True,
            "prior_score_record_bytes_unchanged": prior_preserved,
            "prior_audit_sha256": auditor.sha256(previous_report_path) if previous_report_path else None,
            "previous_factuality_verification_unchanged": factuality_report_hash is not None,
            "previous_factuality_audit_sha256": factuality_report_hash,
            "prefix_status_sha256": auditor.sha256(status_path),
            "prefix_audit_sha256": auditor.sha256(folder / "prefix_audit.json"),
            "source_sha256": manifest["source_sha256"],
            "split_auditor_sha256": auditor.sha256(Path(auditor.__file__)),
            "entrypoint_sha256": auditor.sha256(Path(__file__)),
            "api_calls": 0, "gpu_allocations": 0, "formal_inputs_modified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=int, required=True)
    parser.add_argument("--previous", type=int)
    args = parser.parse_args()
    output = RUN / f"verification_smfa_retain_{args.profiles}.json"
    auditor.require(not output.exists(), "Refusing to replace an existing independent audit")
    result = audit(args.profiles, args.previous)
    auditor.write_json(output, result)
    print(json.dumps({"output": str(output), "state": result["state"],
                      "profiles": result["profiles"], "judge_scores_verified": result["judge_scores_verified"]}))
