"""Judge a missing metric while the full scorer finishes a failed fact batch."""

import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

import audit_final as audit
import audit_smfa_prefix as prefix
import early_smfa_queue as queue
import early_original_forget as helper
import score_smfa_tail as tail

score = tail.score
PROFILES = 625
SENTINEL = 11348
FIELD = "Meaningful Score"
WORK = audit.RUN / "parallel_smfa_meaningful_625"
STATUS = audit.RUN / "parallel_smfa_meaningful_625_status.json"
DEST = tail.DEST


def guard(expected=None):
    queue.verify_sources()
    state = audit.read_json(audit.RUN / "status.json")
    audit.require(state["stage"] == "score_SMFA" and state["state"] in ("running", "failed"),
                  "The formal scorer is no longer in the inspected fact batch")
    if expected is not None:
        audit.require((state["pid"], state["child_pid"]) == (expected["pid"], expected["child_pid"]),
                      "The formal scoring process changed")
    if state["state"] == "running":
        audit.require(audit.live(state["pid"]) and audit.live(state["child_pid"]),
                      "Formal scoring processes exited without a terminal status")
    else:
        audit.require(not audit.live(state["pid"]) and not audit.live(state["child_pid"]),
                      "A failed formal scorer still has a writer")
    cache = DEST / "retain_generation_results_factuality.jsonl"
    found = []
    for line in cache.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            continue
        row = json.loads(line)
        if row["index"] == SENTINEL:
            found.append(row)
    audit.require(len(found) == 1 and found[0]["result"].get("Factuality Score") is None,
                  "The inspected failure no longer blocks the formal meaningfulness stage")
    # The frozen scorer raises after this fact batch if any score is null.
    audit.require(not (DEST / "forget_generation_results_meaningful_score.json").exists(),
                  "The formal scorer may have advanced to meaningfulness")
    early = audit.read_json(queue.STATUS)
    audit.require(early["state"] == "handed_off" and not audit.live(early["pid"])
                  and not audit.live(early["child_pid"]), "An early scoring writer remains")
    return state, found[0]


def publish(questions, before, pipeline):
    guard(pipeline)
    path = WORK / "retain_generation_results_meaningful.jsonl"
    raw = prefix.records(path)
    valid = helper.verified_cache(path, questions, FIELD)
    audit.require(all(row["result"]["Justification"].strip() for row in valid.values()),
                  "A meaningfulness result has no justification")
    audit.require(set(raw) == set(range(len(questions))), "The local metric has missing or duplicate records")
    shared = DEST / path.name
    current, existing = tail.valid_shared(shared, FIELD)
    audit.require(current == before, "The canonical meaningfulness cache changed")
    audit.require(all(existing[index] == valid[index] for index in existing.keys() & valid.keys()),
                  "A reused meaningfulness score differs")
    additions = sorted(valid.keys() - existing.keys())
    expected = before + b"".join(raw[index] for index in additions)
    shutil.copy2(shared, WORK / ("previous_shared_" + shared.name))
    guard(pipeline)
    audit.require(shared.read_bytes() == before, "The canonical cache changed before publication")
    temporary = shared.with_suffix(".parallel_meaningful.tmp")
    with temporary.open("xb") as handle:
        handle.write(expected)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(shared)
    after, actual = tail.valid_shared(shared, FIELD)
    audit.require(after == expected and all(actual[index] == row for index, row in valid.items()),
                  "Parallel meaningfulness publication differs")
    return {"added_scores": len(additions), "local_valid_scores": len(valid),
            "local_null_indexes_excluded": sorted(raw.keys() - valid.keys()),
            "previous_scores_preserved": len(existing), "previous_record_bytes_preserved": True,
            "local_cache_sha256": audit.sha256(path), "shared_sha256": audit.sha256(shared)}


def main():
    os.umask(0o077)
    with (audit.RUN / "parallel_smfa_meaningful_625.lock").open("a") as own_lock, \
            (audit.RUN / "early_smfa.lock").open("a") as worker_lock:
        for lock in (own_lock, worker_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline, sentinel = guard()
        manifest = audit.read_json(audit.RUN / "manifest.json")
        audit.require(sys.prefix == manifest["conda_prefix"], "Use the recorded Conda environment")
        source = audit.RUN / "early_smfa_retain_625/retain_generation_results.json"
        questions = audit.read_json(source)["Generation_Questions"]
        full = audit.read_json(DEST / source.name)["Generation_Questions"]
        audit.require(len(questions) == PROFILES * 19 and full[:len(questions)] == questions,
                      "The partial input differs from the full generation")
        full_audit = audit.read_json(audit.RUN / "verification_full_generation_rouge.json")
        audit.require(full_audit["state"] == "passed"
                      and full_audit["sha256"][str((DEST / source.name).relative_to(audit.ROOT))]
                      == audit.sha256(DEST / source.name), "Full generation is not independently verified")
        audit.require(sentinel["digest"] == score.digest(full[SENTINEL]), "Null score digest differs")
        shared = DEST / "retain_generation_results_meaningful.jsonl"
        before, existing = tail.valid_shared(shared, FIELD)
        audit.require(set(existing) == set(range(500 * 19)) | set(range(850 * 19, 950 * 19)),
                      "Unexpected starting meaningfulness coverage")
        if WORK.exists():
            previous = audit.read_json(STATUS)
            audit.require(previous["state"] == "failed" and not audit.live(previous["pid"])
                          and "Unknown scheme for proxy URL" in previous.get("error", ""),
                          "Existing parallel metric run is not the inspected startup failure")
            audit.require((WORK / source.name).read_bytes() == source.read_bytes(),
                          "Existing parallel input differs")
            cache = WORK / shared.name
            seed = prefix.records(shared)
            expected = b"".join(seed[i] for i in sorted(seed) if i < len(questions))
            audit.require(cache.read_bytes() == expected, "Existing parallel cache changed")
        else:
            WORK.mkdir()
            shutil.copy2(source, WORK / source.name)
            seed = prefix.records(shared)
            (WORK / shared.name).write_bytes(b"".join(seed[i] for i in sorted(seed) if i < len(questions)))
        state = {"pid": os.getpid(), "state": "running", "stage": "meaningful", "started_at": audit.now(),
                 "scope": "SMFA retain first 625 profiles, meaningfulness only; full experiment remains pending",
                 "profiles": PROFILES, "questions": len(questions), "workers": 4, "reused_scores": 9500,
                 "formal_controller_pid": pipeline["pid"], "formal_scorer_pid": pipeline["child_pid"],
                 "input_sha256": audit.sha256(source), "starting_shared_sha256": audit.sha256(shared),
                 "entrypoint_sha256": audit.sha256(Path(__file__)), "source_sha256": manifest["source_sha256"]}
        audit.write_json(STATUS, state)
        try:
            module = importlib.import_module("eval_meaningful")
            audit.require(module.client is not None, "Judge credential is absent")
            module.client = module.client.with_options(timeout=120, max_retries=2)
            try:
                score.score_file(module, WORK / source.name, FIELD, "meaningful", 4)
            except RuntimeError as error:
                path = WORK / shared.name
                rows = prefix.records(path)
                valid = helper.verified_cache(path, questions, FIELD)
                audit.require(set(rows) == set(range(len(questions))) and len(valid) < len(questions),
                              "The parallel scorer failed for an uninspected reason")
                audit.require(str(error) == f"{len(questions) - len(valid)} questions have no {FIELD}; resume to retry them",
                              "The parallel scorer failed outside normal null-score handling")
                state["local_scoring_error"] = str(error)
            publication = publish(questions, before, pipeline)
            state.update(state="complete", stage="complete", published=True, publication=publication,
                         metric_complete=publication["local_valid_scores"] == len(questions),
                         completed_at=audit.now(), process_signals_sent=0)
            audit.write_json(STATUS, state)
            print(json.dumps({"state": "complete", "publication": publication}), flush=True)
        except BaseException as error:
            state.update(state="failed", error=str(error), failed_at=audit.now())
            audit.write_json(STATUS, state)
            raise


if __name__ == "__main__":
    main()
