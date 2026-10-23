"""Independently audit the complete Original comparison as soon as it closes."""

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import audit_final as auditor

RUN = auditor.RUN


def audit_original():
    import pandas as pd
    import torch

    torch.set_num_threads(2)
    manifest = auditor.read_json(RUN / "manifest.json")
    auditor.require(sys.prefix == manifest["conda_prefix"], "Activate the recorded Conda environment")
    for relative, expected in manifest["source_sha256"].items():
        auditor.require(auditor.sha256(auditor.ROOT / relative) == expected, f"Experiment source changed: {relative}")
    controller = auditor.read_json(RUN / "parallel_original_status.json")
    auditor.require(controller["state"] == "complete" and controller.get("promoted") is True,
                    "Complete verified Original results have not been promoted")
    auditor.require(controller["completed_stages"] == ["evaluate_Original", "score_Original"],
                    "Original controller did not complete generation and judging")
    folder = Path(controller["destination"])
    auditor.require(folder == auditor.EVALUATION / "Original", "Original results are in an unexpected location")
    stored_path = folder / "Original_final_evaluation_results.json"
    stored = auditor.read_json(stored_path)
    stored_hash = auditor.sha256(stored_path)
    data_audit = auditor.read_json(RUN / "data_audit.json")
    expected_data = {item["path"]: item["sha256"] for item in data_audit["files"]}
    splits = {}
    for split, group, profiles, questions in (("forget", "Forget Set Results", 50, 950),
                                             ("retain", "Retain Set (shared dataset) Results", 950, 18050)):
        relative = f"eval_data/{split}_5/train-00000-of-00001.parquet"
        dataset = auditor.REPO / "data/S-MLLMUn-data" / relative
        auditor.require(auditor.sha256(dataset) == expected_data[relative], "Evaluation dataset changed")
        splits[split] = auditor.audit_split(folder, split, pd.read_parquet(dataset))
        measured = splits[split]
        auditor.require((measured["profiles"], measured["questions"]) == (profiles, questions),
                        "Original full evaluation coverage differs")
        for kind, rouge_label in zip(auditor.KINDS, auditor.ROUGE_LABELS):
            auditor.equal_mean(stored[group]["generation"][f"Average ROUGE-L ({rouge_label})"],
                               measured["rougeL"][kind], f"Original/{split}/{kind} ROUGE")
        measured["data_sha256"] = expected_data[relative]
        print(f"Independently verified Original/{split}: {questions} answers, {questions * 2} scores", flush=True)
    for split, measured in splits.items():
        for name, expected in measured["sha256"].items():
            auditor.require(auditor.sha256(folder / name) == expected, f"Audited {split} file changed")
    auditor.require(auditor.sha256(stored_path) == stored_hash, "ROUGE report changed during audit")
    reference = auditor.read_json(RUN / "paper_reference.json")
    overall = auditor.aggregate(splits)
    paper = reference["Original"]["overall"]
    return {"state": "passed", "checked_at": auditor.now(),
            "scope": "Complete Original comparison only; formal SMFA experiment remains pending",
            "generated_answers_verified": 19000, "judge_scores_verified": 38000,
            "splits": splits, "overall": overall, "paper_overall": paper,
            "difference_from_paper": {key: overall[key] - paper[key] for key in overall},
            "rouge_summary_sha256": stored_hash,
            "controller_status_sha256": auditor.sha256(RUN / "parallel_original_status.json"),
            "paper_reference_sha256": auditor.sha256(RUN / "paper_reference.json"),
            "source_sha256": manifest["source_sha256"],
            "split_auditor_sha256": auditor.sha256(Path(auditor.__file__)),
            "auditor_sha256": auditor.sha256(Path(__file__)), "api_calls": 0,
            "notes": ["This is a full Original baseline audit, not final SMFA results.",
                      "The Qwen-Plus alias and its judging behavior can vary over time."]}


def write_report(result):
    destination = auditor.ROOT / "ORIGINAL_BASELINE_AUDIT.md"
    auditor.require(not destination.exists(), "Refusing to replace an existing baseline report")
    lines = ["# Complete Original Baseline Audit", "", result["scope"], "",
             "All 19,000 generated answers and 38,000 judge scores passed complete coverage,",
             "question/image identity, score validity, independent ROUGE and aggregation checks.", "",
             "| Metric | Measured | Paper | Difference |", "| --- | --- | --- | --- |"]
    for key, value in result["overall"].items():
        lines.append(f"| {key} | {value:.6f} | {result['paper_overall'][key]:.6f} | "
                     f"{result['difference_from_paper'][key]:+.6f} |")
    lines += ["", "ROUGE-L is on a 0–1 scale. Factuality and meaningfulness are on a 0–10 scale.",
              "Qwen-Plus judging can vary with the current model alias.",
              "SMFA training, generation, judging and final audits remain separate requirements.",
              "", "Evidence: `smfa_run/original_full_audit.json`."]
    temporary = destination.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(destination)


def main(watch):
    os.umask(0o077)
    with (RUN / "original_full_audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hashes = {str(Path(module.__file__)): auditor.sha256(Path(module.__file__))
                  for module in (auditor, sys.modules[__name__])}
        state = {"pid": os.getpid(), "state": "running", "stage": "waiting_for_original",
                 "started_at": auditor.now(), "source_sha256": hashes, "api_calls": 0}
        status = RUN / "original_full_audit_status.json"
        auditor.write_json(status, state)
        try:
            if watch:
                while True:
                    controller = auditor.read_json(RUN / "parallel_original_status.json")
                    if controller["state"] == "complete":
                        if not auditor.live(controller["pid"]):
                            break
                    else:
                        auditor.require(controller["state"] == "running" and auditor.live(controller["pid"]),
                                        "Original controller failed or stopped; inspect before auditing")
                    time.sleep(30)
            for path, expected in hashes.items():
                auditor.require(auditor.sha256(Path(path)) == expected, "Baseline audit source changed")
            state.update(stage="auditing_full_original", checked_at=auditor.now())
            auditor.write_json(status, state)
            destination = RUN / "original_full_audit.json"
            auditor.require(not destination.exists(), "Refusing to replace an existing baseline audit")
            result = audit_original()
            auditor.write_json(destination, result)
            write_report(result)
            state.update(state="complete", stage="complete", completed_at=auditor.now(),
                         audit_sha256=auditor.sha256(destination))
            auditor.write_json(status, state)
            print("Complete Original comparison independently audited; SMFA remains pending", flush=True)
        except BaseException as error:
            state.update(state="failed", error=str(error), failed_at=auditor.now())
            auditor.write_json(status, state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        os.umask(0o077)
        with (RUN / "original_full_audit.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--watch"],
                                       env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Launched complete Original audit watcher PID {process.pid}")
    else:
        main(args.watch)
