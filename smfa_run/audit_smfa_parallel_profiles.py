"""Audit an immutable snapshot of the parallel SMFA suffix without inference."""

import hashlib
import json
from pathlib import Path
import sys

import audit_final as auditor

ROOT = auditor.ROOT
RUN = auditor.RUN
DEST = auditor.EVALUATION / "SMFA"


def questions(path, rows):
    output = auditor.read_json(path)["Generation_Questions"]
    expected = [(kind, row, item) for _, row in rows.iterrows()
                for kind, column in zip(auditor.KINDS, auditor.COLUMNS)
                for item in row[column]]
    auditor.require(len(output) == len(expected) == 19, "Profile coverage differs")
    for question, (kind, row, item) in zip(output, expected):
        auditor.require((question["question type"], question["question"], question["ground_truth"])
                        == (kind, item["Question"], item["Answer"]), "Question identity differs")
        answer = question.get("generated_answer")
        auditor.require(isinstance(answer, str) and answer.strip(), "Empty generated answer")
        if kind != "text_memory":
            auditor.require((question.get("image_id"), question.get("image_type"))
                            == (row["ID"], item["Image"]), "Image identity differs")
    return output


def main():
    import pandas as pd

    manifest = auditor.read_json(RUN / "manifest.json")
    auditor.require(sys.prefix == manifest["conda_prefix"], "Use the recorded Conda environment")
    for name, expected in manifest["source_sha256"].items():
        auditor.require(auditor.sha256(ROOT / name) == expected, f"Frozen source changed: {name}")
    status_path = RUN / "parallel_smfa_profiles_status.json"
    snapshot = status_path.read_bytes()
    state = json.loads(snapshot)
    auditor.require(state["state"] in ("running", "complete"), "Suffix worker failed")
    auditor.require(state["generation"] == {"max_new_tokens": 100, "do_sample": False},
                    "Decoding settings differ")
    auditor.require(state["physical_gpu"] == "0" and state["api_calls"] == 0,
                    "Worker configuration differs")
    auditor.require(auditor.sha256(RUN / "parallel_smfa_profiles.py") == state["entrypoint_sha256"],
                    "Suffix worker source changed")
    auditor.require(auditor.sha256(auditor.REPO / "eval.py") == state["official_evaluator_sha256"],
                    "Official evaluator changed")
    dataset = auditor.REPO / "data/S-MLLMUn-data/eval_data/retain_5/train-00000-of-00001.parquet"
    auditor.require(auditor.sha256(dataset) == state["data_sha256"], "Evaluation dataset changed")
    rows = pd.read_parquet(dataset)
    auditor.require(len(rows) == state["total_profiles"] == 950, "Dataset coverage differs")

    probe_hashes = {}
    auditor.require([p["profile"] for p in state["probes"]] == [0, 125, 250], "Missing GPU probes")
    for probe in state["probes"]:
        index = probe["profile"]
        baseline = DEST / ".profiles/retain" / f"{index:04d}" / "retain_generation_results.json"
        generated = RUN / "parallel_smfa_profiles/probes" / f"{index:04d}" / baseline.name
        auditor.require(auditor.sha256(baseline) == probe["baseline_sha256"], "Probe baseline changed")
        auditor.require(auditor.sha256(generated) == probe["probe_sha256"] == probe["baseline_sha256"],
                        "Cross-GPU serialized answers differ")
        auditor.require(questions(generated, rows.iloc[index:index + 1])
                        == questions(baseline, rows.iloc[index:index + 1]), "Cross-GPU answers differ")
        auditor.require(probe["questions"] == 19 and probe["exact_output_equal"]
                        and probe["cpu_cuda_rng_unchanged"], "Probe attestation differs")
        probe_hashes[str(generated.relative_to(ROOT))] = auditor.sha256(generated)

    published = state["published_profiles"]
    indexes = [item["profile"] for item in published]
    auditor.require(indexes == list(range(949, 949 - len(published), -1)), "Suffix indexes differ")
    hashes = {}
    for item in published:
        index = item["profile"]
        auditor.require(index - item["main_completed_profiles_at_publication"]
                        >= state["minimum_profile_gap"] >= 100, "Publication gap differs")
        path = DEST / ".profiles/retain" / f"{index:04d}" / "retain_generation_results.json"
        auditor.require(auditor.sha256(path) == item["sha256"], "Published profile changed")
        auditor.require(len(questions(path, rows.iloc[index:index + 1])) == item["questions"] == 19,
                        "Published profile coverage differs")
        auditor.require(item["cpu_cuda_rng_unchanged"], "RNG attestation differs")
        hashes[str(path.relative_to(ROOT))] = item["sha256"]
    auditor.require(state["published_questions"] == 19 * len(published), "Published total differs")
    for name, metadata in state["model_files_before"].items():
        stat = Path(name).stat()
        auditor.require([stat.st_size, stat.st_mtime_ns] == metadata, "Checkpoint metadata changed")
    for name, expected in {**hashes, **probe_hashes}.items():
        auditor.require(auditor.sha256(ROOT / name) == expected, "Profile changed during audit")

    progress = auditor.read_json(DEST / "progress.json")
    auditor.require(progress["split"] == "retain", "Primary progress split differs")
    prefix = progress["completed_profiles"]
    unique = set(range(prefix)) | set(indexes)
    result = {"state": "passed", "checked_at": auditor.now(),
              "scope": "Published parallel retain suffix snapshot only; full experiment remains pending",
              "worker_state": state["state"], "worker_status_snapshot": state,
              "worker_status_snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
              "published_profiles_verified": len(published), "published_answers_verified": 19 * len(published),
              "probe_answers_verified": 57, "profile_sha256": hashes, "probe_sha256": probe_hashes,
              "primary_prefix_profiles_at_snapshot": prefix,
              "unique_retain_profiles_at_snapshot": len(unique),
              "unique_retain_answers_at_snapshot": 19 * len(unique),
              "checkpoint_metadata_unchanged": True, "source_sha256": manifest["source_sha256"],
              "entrypoint_sha256": auditor.sha256(Path(__file__)),
              "api_calls": 0, "gpu_allocations": 0, "formal_inputs_modified": False}
    stamp = auditor.now().replace(":", "").replace(".", "_")
    destination = RUN / f"verification_smfa_parallel_profiles_{stamp}.json"
    auditor.require(not destination.exists(), "Existing snapshot audit")
    auditor.write_json(destination, result)
    print(json.dumps({"output": str(destination), **{k: result[k] for k in (
        "state", "published_profiles_verified", "published_answers_verified", "probe_answers_verified",
        "unique_retain_answers_at_snapshot", "api_calls", "gpu_allocations")}}))


if __name__ == "__main__":
    main()
