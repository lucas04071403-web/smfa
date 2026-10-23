"""Close early judging and preserve valid scores before full generation ends."""

from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time

import audit_final as auditor
import audit_smfa_prefix as prefix
import early_smfa_queue as queue
import early_original_forget as helper
import score_smfa_tail as tail


def identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return fields[0], fields[19]


def generation_guard():
    state = auditor.read_json(auditor.RUN / "status.json")
    auditor.require(state["state"] == "running" and state["stage"] == "evaluate_SMFA"
                    and queue.live(state["pid"]) and queue.live(state["child_pid"]),
                    "Formal generation is not running")
    progress = auditor.read_json(tail.DEST / "progress.json")
    auditor.require(progress["split"] == "retain" and progress["completed_profiles"] <= 708
                    and not (tail.DEST / "retain_generation_results.json").exists(),
                    "The full scorer may be near its handoff; leave partial scores isolated")
    return state, progress


def main():
    os.umask(0o077)
    queue.verify_sources()
    main_state, progress = generation_guard()
    auditor.require(progress["completed_profiles"] >= 700, "Generation has not reached the handoff window")
    scheduler = auditor.read_json(queue.STATUS)
    auditor.require(scheduler["state"] == "running" and scheduler["stage"] == "judging"
                    and scheduler["target"] == ["retain", 625], "Unexpected active early batch")
    worker_path = auditor.RUN / "early_smfa_retain_625_status.json"
    worker = auditor.read_json(worker_path)
    auditor.require(worker["pid"] == scheduler["child_pid"] and worker["state"] == "running"
                    and worker["stage"] == "factuality", "Unexpected early worker")
    evidence = auditor.RUN / ("smfa_partial_handoff_" + auditor.now().replace(":", "").replace(".", "_"))
    evidence.mkdir()
    identities = {}
    for state in (scheduler, worker):
        pid = state["pid"]
        auditor.require(pid not in (main_state["pid"], main_state["child_pid"]), "Refusing to signal generation")
        identities[pid] = identity(pid)
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        auditor.require(any(argument.endswith(b"/early_smfa_queue.py") for argument in arguments),
                        "The target is not the expected scoring script")
        auditor.require(identities[pid][0] not in ("Z", "T", "t"), "Scoring process is not active")
    for source in (queue.STATUS, worker_path):
        shutil.copy2(source, evidence / source.name)
    generation_guard()
    for pid, expected in identities.items():
        current = identity(pid)
        auditor.require(current[1] == expected[1] and current[0] not in ("Z", "T", "t"),
                        "Scoring process identity changed or is inactive")
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while any(queue.live(pid) for pid in identities) and time.monotonic() < deadline:
        time.sleep(0.1)
    auditor.require(all(not queue.live(pid) for pid in identities), "An early scoring writer remains live")
    with ExitStack() as stack:
        for name in ("early_smfa_queue.lock", "early_smfa.lock"):
            lock = stack.enter_context((auditor.RUN / name).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generation_guard()
        work = auditor.RUN / "early_smfa_retain_625"
        source = work / "retain_generation_results.json"
        auditor.require(auditor.sha256(source) == worker["input_sha256"], "Partial generation changed")
        questions = auditor.read_json(source)["Generation_Questions"]
        auditor.require(len(questions) == 625 * 19, "Partial question coverage differs")
        source_questions = []
        for index in range(625):
            path = tail.DEST / ".profiles/retain" / f"{index:04d}" / source.name
            source_questions.extend(auditor.read_json(path)["Generation_Questions"])
        auditor.require(source_questions == questions, "Partial input differs from source profiles")
        publications, cache_hashes = {}, {}
        for _, metric, field in tail.METRICS:
            cache = work / f"retain_generation_results_{metric}.jsonl"
            if not cache.exists():
                continue
            local = prefix.records(cache)
            verified = helper.verified_cache(cache, questions, field)
            auditor.require(all(item["result"]["Justification"].strip() for item in verified.values()),
                            "A partial score has no justification")
            auditor.require(all(0 <= index < len(questions) for index in local), "Partial cache index differs")
            shared = tail.DEST / cache.name
            before, existing = tail.valid_shared(shared, field)
            auditor.require(all(existing[i] == verified[i] for i in existing.keys() & verified.keys()),
                            "A partial score conflicts with the canonical cache")
            additions = sorted(verified.keys() - existing.keys())
            data = before + b"".join(local[i] for i in additions)
            shutil.copy2(cache, evidence / ("local_" + cache.name))
            shutil.copy2(shared, evidence / ("previous_shared_" + cache.name))
            generation_guard()
            auditor.require(shared.read_bytes() == before, "The shared cache changed")
            temporary = shared.with_suffix(".handoff.tmp")
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(shared)
            after, actual = tail.valid_shared(shared, field)
            auditor.require(after == data and all(actual[i] == row for i, row in verified.items()),
                            "Partial score publication differs")
            publications[metric] = {"added_scores": len(additions), "local_valid_scores": len(verified),
                                    "previous_scores_preserved": len(existing),
                                    "shared_sha256": auditor.sha256(shared),
                                    "previous_record_bytes_preserved": after.startswith(before),
                                    "unusable_local_scores_excluded": len(local) - len(verified)}
            cache_hashes[str(cache.relative_to(auditor.ROOT))] = auditor.sha256(cache)
        result = {"state": "passed", "checked_at": auditor.now(), "api_calls": 0,
                  "purpose": "Avoid overlapping early and full scoring while preserving closed valid records",
                  "generation_pids_untouched": [main_state["pid"], main_state["child_pid"]],
                  "scoring_pids_stopped": list(identities), "publications": publications,
                  "closed_local_cache_sha256": cache_hashes,
                  "inflight_tasks_disclosure": "Up to four in-flight scoring tasks may have called the API without recording a response; the exact count is unknown.",
                  "entrypoint_sha256": auditor.sha256(Path(__file__))}
        auditor.write_json(evidence / "handoff.json", result)
        for state, path in ((scheduler, queue.STATUS), (worker, worker_path)):
            state.update(state="handed_off", stage="awaiting_full_smfa_scoring", handed_off_at=auditor.now(),
                         partial_batch=True, handoff_evidence=str(evidence.relative_to(auditor.ROOT)))
            auditor.write_json(path, state)
        print(json.dumps({"state": "handed_off", "evidence": str(evidence), "publications": publications,
                          "api_calls": 0, "generation_untouched": True}))


if __name__ == "__main__":
    main()
