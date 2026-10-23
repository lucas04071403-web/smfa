"""Independently replay the closed partial-score publication without API calls."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import audit_final as audit


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def records(data, questions, field):
    audit.require(not data or data.endswith(b"\n"), "Incomplete score record")
    result = {}
    for line in data.splitlines(keepends=True):
        row = json.loads(line)
        index = row["index"]
        audit.require(type(index) is int and 0 <= index < len(questions) and index not in result,
                      "Invalid or duplicate score index")
        question = questions[index]
        audit.require(row["digest"] == digest(question), "Question digest differs")
        value = row["result"]
        audit.require(type(value.get(field)) is int and 0 <= value[field] <= 10, "Invalid score")
        audit.require(isinstance(value.get("Justification"), str) and value["Justification"].strip(),
                      "Missing justification")
        for destination, source in (("Question", "question"), ("Generated Answer", "generated_answer"),
                                    ("Ground Truth", "ground_truth"), ("Question Type", "question type"),
                                    ("Image_ID", "image_id"), ("image_type", "image_type")):
            audit.require(value.get(destination) == question.get(source), "Score identity differs")
        result[index] = line
    return result


def main(evidence):
    import pandas as pd

    manifest = audit.read_json(audit.RUN / "manifest.json")
    audit.require(sys.prefix == manifest["conda_prefix"], "Use the recorded Conda environment")
    for name, expected in manifest["source_sha256"].items():
        audit.require(audit.sha256(audit.ROOT / name) == expected, "Frozen source changed")
    handoff = audit.read_json(evidence / "handoff.json")
    audit.require(handoff["state"] == "passed" and handoff["api_calls"] == 0, "Handoff did not pass")
    audit.require(audit.sha256(audit.RUN / "handoff_smfa_partial_scores.py") == handoff["entrypoint_sha256"],
                  "Handoff entrypoint changed")
    dest = audit.EVALUATION / "SMFA"
    source = dest / "retain_generation_results.json"
    assembly_path = audit.RUN / "verification_smfa_retain_assembly.json"
    assembly = audit.read_json(assembly_path)
    audit.require(assembly["state"] == "passed" and assembly["answers_verified"] == 18050,
                  "Full retain assembly is not independently verified")
    audit.require(assembly["sha256"][str(source.relative_to(audit.ROOT))] == audit.sha256(source),
                  "Audited generation changed")
    questions = audit.read_json(source)["Generation_Questions"]
    dataset = audit.REPO / "data/S-MLLMUn-data/eval_data/retain_5/train-00000-of-00001.parquet"
    rows = pd.read_parquet(dataset)
    expected = [(kind, row, item) for _, row in rows.iterrows()
                for kind, column in zip(audit.KINDS, audit.COLUMNS) for item in row[column]]
    audit.require(len(questions) == len(expected) == 18050, "Retain coverage differs")
    for question, (kind, row, item) in zip(questions, expected):
        audit.require((question["question type"], question["question"], question["ground_truth"])
                      == (kind, item["Question"], item["Answer"]), "Dataset identity differs")
        if kind != "text_memory":
            audit.require((question.get("image_id"), question.get("image_type"))
                          == (row["ID"], item["Image"]), "Image identity differs")
    work = audit.RUN / "early_smfa_retain_625"
    worker = audit.read_json(evidence / "early_smfa_retain_625_status.json")
    partial_source = work / source.name
    audit.require(audit.sha256(partial_source) == worker["input_sha256"], "Partial input changed")
    audit.require(audit.read_json(partial_source)["Generation_Questions"] == questions[:625 * 19],
                  "Partial answers differ from final assembly")
    verified = {}
    for metric, publication in handoff["publications"].items():
        field = {"factuality": "Factuality Score", "meaningful": "Meaningful Score"}[metric]
        filename = f"retain_generation_results_{metric}.jsonl"
        local_path = evidence / ("local_" + filename)
        prior_path = evidence / ("previous_shared_" + filename)
        local_data, prior_data = local_path.read_bytes(), prior_path.read_bytes()
        current_data = (dest / filename).read_bytes()
        closed_path = work / filename
        audit.require(audit.sha256(closed_path)
                      == handoff["closed_local_cache_sha256"][str(closed_path.relative_to(audit.ROOT))],
                      "Closed partial cache changed")
        audit.require(local_data == closed_path.read_bytes(), "Closed snapshot differs")
        local = records(local_data, questions[:625 * 19], field)
        prior = records(prior_data, questions, field)
        audit.require(set(prior) == set(range(500 * 19)) | set(range(850 * 19, 950 * 19)),
                      "Previous prefix/tail coverage differs")
        audit.require(all(json.loads(local[i]) == json.loads(prior[i]) for i in local.keys() & prior.keys()),
                      "Reused score changed")
        additions = sorted(local.keys() - prior.keys())
        published = prior_data + b"".join(local[i] for i in additions)
        audit.require(hashlib.sha256(published).hexdigest() == publication["shared_sha256"],
                      "Published snapshot hash differs")
        audit.require(current_data.startswith(published), "Canonical publication bytes changed")
        audit.require((len(additions), len(local), len(prior))
                      == (publication["added_scores"], publication["local_valid_scores"],
                          publication["previous_scores_preserved"]), "Publication counts differ")
        verified[metric] = {"new_scores_verified": len(additions), "local_scores_verified": len(local),
                            "previous_record_bytes_preserved": len(prior),
                            "published_snapshot_sha256": hashlib.sha256(published).hexdigest(),
                            "current_cache_may_have_additional_full_run_records": True}
    for pid in handoff["scoring_pids_stopped"]:
        audit.require(not audit.live(pid), "An early scorer is still live")
    pipeline = audit.read_json(audit.RUN / "status.json")
    audit.require(pipeline["pid"] == handoff["generation_pids_untouched"][0]
                  and audit.live(pipeline["pid"]), "Formal controller changed or stopped")
    output = evidence / "verification.json"
    audit.require(not output.exists(), "Refusing to replace independent evidence")
    result = {"state": "passed", "checked_at": audit.now(), "publications": verified,
              "scope": "Partial handoff and preservation only; full judging remains pending",
              "handoff_sha256": audit.sha256(evidence / "handoff.json"),
              "assembly_audit_sha256": audit.sha256(assembly_path), "data_sha256": audit.sha256(dataset),
              "entrypoint_sha256": audit.sha256(Path(__file__)), "api_calls": 0, "gpu_allocations": 0,
              "source_sha256": manifest["source_sha256"],
              "inflight_tasks_disclosure": handoff["inflight_tasks_disclosure"]}
    audit.write_json(output, result)
    print(json.dumps({"output": str(output), "state": "passed", "publications": verified, "api_calls": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    main(parser.parse_args().evidence.resolve())
