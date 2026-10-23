"""Resume closed Original scoring from complete caches without loading a model."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import audit_final as auditor
import early_original_forget as helper
import parallel_original as original
import recover_original_full as recovery

RUN, ROOT = auditor.RUN, auditor.ROOT
runner = original.runner


def complete_cache(path, questions, field):
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.splitlines()]
    indexes = [row["index"] for row in rows]
    auditor.require(data.endswith(b"\n") and len(indexes) == len(questions)
                    and all(type(i) is int for i in indexes)
                    and set(indexes) == set(range(len(questions))), "Cache coverage is incomplete or duplicated")
    records = helper.verified_cache(path, questions, field)
    auditor.require(len(records) == len(questions)
                    and all(row["result"]["Justification"].strip() for row in records.values()),
                    "Cache has invalid or missing scores")
    return auditor.sha256(path)


def main():
    os.umask(0o077)
    with (RUN / "parallel_original.lock").open("a") as lock, \
            (RUN / "early_original.lock").open("a") as early_lock:
        for handle in (lock, early_lock):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = auditor.read_json(original.STATUS)
        recovery.require_closed(previous)
        manifest = auditor.read_json(RUN / "manifest.json")
        auditor.require(sys.prefix == manifest["conda_prefix"], "Activate the recorded Conda environment")
        for relative, expected in manifest["source_sha256"].items():
            auditor.require(auditor.sha256(ROOT / relative) == expected, f"Formal source changed: {relative}")
        auditor.require(Path(previous["staging"]) == original.STAGING
                        and Path(previous["destination"]) == original.DESTINATION
                        and not original.DESTINATION.exists(), "Unexpected Original output location")
        auditor.require(auditor.read_json(RUN / "status.json")["stage"]
                        not in ("evaluate_Original", "score_Original"), "Formal Original evaluation may be active")
        generation = auditor.read_json(RUN / "original_generation_audit.json")
        auditor.require(generation["state"] == "passed" and generation["generated_answers_verified"] == 19000,
                        "Complete Original generation is not audited")
        protected, counts = {}, {}
        for split, count in (("forget", 950), ("retain", 18050)):
            source = original.STAGING / f"{split}_generation_results.json"
            digest = auditor.sha256(source)
            auditor.require(digest == generation["splits"][split]["generation_sha256"], "Generated answers changed")
            protected[source] = digest
            questions = auditor.read_json(source)["Generation_Questions"]
            auditor.require(len(questions) == count, "Question count differs")
            for suffix, field in (("factuality", "Factuality Score"), ("meaningful", "Meaningful Score")):
                cache = original.STAGING / f"{split}_generation_results_{suffix}.jsonl"
                protected[cache] = complete_cache(cache, questions, field)
                counts[f"{split}_{suffix}"] = count
        command = list(manifest["commands"]["score_Original"])
        command[command.index("--input_folder") + 1] = str(original.STAGING)
        auditor.require(command == previous["commands"]["score_Original"], "Full scoring command changed")
        stamp = auditor.now().replace(":", "").replace("+0000", "Z")
        evidence = RUN / f"original_full_score_resume_{stamp}"
        evidence.mkdir()
        for path in (original.STATUS, RUN / "parallel_original.log", RUN / "parallel_score_Original.log"):
            shutil.copy2(path, evidence / path.name)
        auditor.write_json(evidence / "provenance.json", {
            "checked_at": auditor.now(), "complete_cached_scores": counts,
            "protected_files_sha256": {str(p.relative_to(ROOT)): h for p, h in protected.items()},
            "source_sha256": manifest["source_sha256"],
            "resume_source_sha256": auditor.sha256(Path(__file__)),
            "note": "Only the unchanged full scorer runs; no model or generation process is started."})
        state = dict(previous)
        for key in ("error", "failed_at", "child_pid"):
            state.pop(key, None)
        state.update(pid=os.getpid(), state="running", stage="score_Original", resumed_at=auditor.now(),
                     stage_started_at=auditor.now(), physical_gpu="",
                     resume_evidence=str(evidence.relative_to(ROOT)))
        auditor.write_json(original.STATUS, state)
        try:
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            with (RUN / "parallel_score_Original.log").open("a") as log:
                child = subprocess.Popen(command, cwd=runner.REPO, env=environment,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                state["child_pid"] = child.pid
                auditor.write_json(original.STATUS, state)
                code = child.wait()
            auditor.require(code == 0, f"Resumed full scorer exited with code {code}")
            for path, expected in protected.items():
                auditor.require(auditor.sha256(path) == expected, "An audited generation or complete cache changed")
            previous_root = runner.EVAL_ROOT
            try:
                runner.EVAL_ROOT = str(original.STAGING.parent)
                runner.check_stage("evaluate_Original")
                runner.check_stage("score_Original")
            finally:
                runner.EVAL_ROOT = previous_root
            auditor.require(not original.DESTINATION.exists(), "Original destination appeared during scoring")
            original.STAGING.rename(original.DESTINATION)
            runner.check_stage("evaluate_Original")
            runner.check_stage("score_Original")
            state.pop("child_pid", None)
            state.update(state="complete", stage="complete", promoted=True,
                         completed_stages=["evaluate_Original", "score_Original"], completed_at=auditor.now())
            auditor.write_json(original.STATUS, state)
            print("Original full scoring and promotion complete; independent baseline audit still required", flush=True)
        except BaseException as error:
            state.update(state="failed", error=str(error), failed_at=auditor.now())
            auditor.write_json(original.STATUS, state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        os.umask(0o077)
        with (RUN / "resume_original_scoring.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        print(f"Launched score-only Original resume PID {process.pid}")
    else:
        main()
