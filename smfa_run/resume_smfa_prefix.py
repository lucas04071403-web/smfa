"""Finish a recovered SMFA prefix after its failed worker has exited."""

import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "smfa_run"
REPO = ROOT / "S-MLLMUn"
sys.path.insert(0, str(ROOT / "reproduce_smfa"))
import run as runner
import score
import audit_final as auditor
import early_original_forget as helper


def sha(path):
    return auditor.sha256(path)


def live(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def main(args):
    manifest = auditor.read_json(RUN / "manifest.json")
    auditor.require(sys.prefix == manifest["conda_prefix"], "Use the recorded Conda environment")
    for relative, expected in manifest["source_sha256"].items():
        auditor.require(sha(ROOT / relative) == expected, f"Frozen source changed: {relative}")
    label = f"early_smfa_{args.split}_{args.profiles}"
    state_path, queue_path = RUN / f"{label}_status.json", RUN / "early_smfa_queue_status.json"
    with (RUN / "early_smfa.lock").open("a") as worker_lock, (RUN / "early_smfa_queue.lock").open("a") as queue_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, queue = auditor.read_json(state_path), auditor.read_json(queue_path)
        auditor.require(state["state"] == "failed" and queue["state"] == "failed",
                        "The failed prefix and queue must be inspected first")
        auditor.require(not live(state["pid"]) and not live(queue["pid"]), "Refusing to resume live processes")
        work = RUN / label
        source = work / f"{args.split}_generation_results.json"
        questions = auditor.read_json(source)["Generation_Questions"]
        auditor.require(len(questions) == args.profiles * 19, "Prefix coverage differs")
        rows = __import__("pandas").read_parquet(REPO / f"data/S-MLLMUn-data/eval_data/{args.split}_5/train-00000-of-00001.parquet")
        rows = rows.iloc[:args.profiles]
        from evaluate import verify_questions
        verify_questions(questions, rows)
        state.update(state="running", stage="meaningful", resumed_at=runner.now(),
                     recovery_evidence="smfa_retain_375_factuality_json_recovery_20261021T180115Z")
        runner.write_json(state_path, state)
        try:
            module = importlib.import_module("eval_meaningful")
            auditor.require(module.client is not None, "Judge credential is absent")
            module.client = module.client.with_options(timeout=120, max_retries=2)
            # Rehydrate the validated 250-profile prefix before asking the API for the suffix.
            canonical = REPO / "eval_result/llava-onevision/split_ratio_5/SMFA"
            meaningful_cache = work / f"{args.split}_generation_results_meaningful.jsonl"
            prior = helper.verified_cache(canonical / meaningful_cache.name, questions, "Meaningful Score")
            if prior:
                meaningful_cache.write_text("".join(json.dumps(prior[index]) + "\n" for index in sorted(prior)))
            state["reused_scores"]["meaningful"] = len(prior)
            runner.write_json(state_path, state)
            score.score_file(module, source, "Meaningful Score", "meaningful", 4)
            state["completed_metrics"] = ["factuality", "meaningful"]
            runner.write_json(state_path, state)
            audit = auditor.audit_split(work, args.split, rows)
            runner.write_json(work / "prefix_audit.json", audit)
            progress = json.loads((REPO / "eval_result/llava-onevision/split_ratio_5/SMFA/progress.json").read_text())
            published = progress["split"] == args.split and progress["completed_profiles"] >= args.profiles
            additions = {}
            if published:
                canonical = REPO / "eval_result/llava-onevision/split_ratio_5/SMFA"
                for metric, field in (("factuality", "Factuality Score"), ("meaningful", "Meaningful Score")):
                    additions[metric] = helper.append_verified_cache(
                        work / f"{args.split}_generation_results_{metric}.jsonl",
                        canonical / f"{args.split}_generation_results_{metric}.jsonl",
                        questions, field)
            state.update(state="complete", stage="complete", completed_at=runner.now(), published=published,
                         appended_scores=additions, audit_sha256=sha(work / "prefix_audit.json"))
            runner.write_json(state_path, state)
            print(json.dumps({"state": state["state"], "published": published,
                              "appended_scores": additions, "questions": len(questions)}))
        except BaseException as error:
            state.update(state="failed", failed_at=runner.now(), error=str(error))
            runner.write_json(state_path, state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("forget", "retain"), required=True)
    parser.add_argument("--profiles", type=int, required=True)
    main(parser.parse_args())
