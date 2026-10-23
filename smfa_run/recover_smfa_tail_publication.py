"""Publish audited tail caches after all failed scoring writers have exited."""

from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil

import audit_final as auditor
import audit_smfa_prefix as prefix
import early_smfa_queue as queue
import score_smfa_tail as tail


def publish(evidence):
    os.umask(0o077)
    with ExitStack() as stack:
        for name in ("early_smfa_queue.lock", "early_smfa.lock", "smfa_tail_850_950.lock"):
            lock = stack.enter_context((auditor.RUN / name).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue.verify_sources()
        scheduler = auditor.read_json(queue.STATUS)
        state = auditor.read_json(tail.STATUS)
        watcher_path = auditor.RUN / "smfa_tail_audit_status.json"
        watcher = auditor.read_json(watcher_path)
        auditor.require(scheduler["state"] == state["state"] == watcher["state"] == "failed",
                        "Inspect failed scoring processes before publication")
        auditor.require(all(not queue.live(item["pid"]) for item in (scheduler, state, watcher)),
                        "A scoring process is still live")
        auditor.require(state.get("published") is False
                        and set(state["completed_metrics"]) == {"factuality", "meaningful"},
                        "Tail caches are not complete and isolated")
        auditor.require(tail.generation_available(), "Full scoring handoff has started")
        auditor.require(auditor.sha256(Path(tail.__file__)) == state["entrypoint_sha256"],
                        "Tail scorer source changed")
        local_report = auditor.RUN / "verification_smfa_tail_850_950_local.json"
        local = auditor.read_json(local_report)
        auditor.require(local["state"] == "passed" and local["judge_scores_verified"] == 3800,
                        "Independent local tail audit is unavailable")
        for name, digest in local["split"]["sha256"].items():
            auditor.require(auditor.sha256(tail.WORK / name) == digest, "Audited tail cache changed")
        questions, hashes = tail.snapshot()
        auditor.require(hashes == state["profile_sha256"]
                        and questions == auditor.read_json(tail.WORK / "retain_generation_results.json")["Generation_Questions"],
                        "Tail source profiles changed")
        closed = auditor.read_json(auditor.RUN / "early_smfa_retain_500_status.json")
        auditor.require(closed["state"] == "complete" and closed.get("published") is True
                        and not queue.live(closed["pid"]), "The recovered prefix is not closed")
        prior_report = auditor.read_json(auditor.RUN / "verification_smfa_retain_500.json")
        auditor.require(prior_report["state"] == "passed" and prior_report["profiles"] == 500,
                        "Independent prefix audit is unavailable")
        evidence.mkdir()
        for source in (queue.STATUS, tail.STATUS, watcher_path):
            shutil.copy2(source, evidence / source.name)
        planned = []
        for _, metric, field in tail.METRICS:
            path = tail.DEST / f"retain_generation_results_{metric}.jsonl"
            before, existing = tail.valid_shared(path, field)
            previous = prefix.records(auditor.RUN / "early_smfa_retain_500" / path.name)
            auditor.require(set(existing) == set(previous) == set(range(9500)),
                            "Shared cache is not exactly the recovered prefix")
            auditor.require(before == b"".join(previous[i] for i in sorted(previous)),
                            "Published prefix record bytes differ")
            mapped = tail.remap(tail.WORK / path.name, questions, field)
            auditor.require(not (existing.keys() & mapped.keys()), "Tail overlaps published scores")
            data = before + b"".join((json.dumps(mapped[i]) + "\n").encode() for i in sorted(mapped))
            shutil.copy2(path, evidence / path.name)
            planned.append((path, field, before, data, mapped))
        publications = {}
        for path, field, before, data, mapped in planned:
            auditor.require(path.read_bytes() == before, "Shared cache changed before publication")
            temporary = path.with_suffix(".recover-tail.tmp")
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            after, actual = tail.valid_shared(path, field)
            auditor.require(after == data and all(actual.get(i) == item for i, item in mapped.items()),
                            "Published tail records differ")
            metric = "factuality" if field == "Factuality Score" else "meaningful"
            publications[metric] = {"added_records": len(mapped), "previous_records": 9500,
                                    "shared_sha256": auditor.sha256(path),
                                    "previous_record_bytes_preserved": after.startswith(before)}
        state.update(state="complete", stage="complete", pid=os.getpid(), published=True,
                     completed_at=auditor.now(), publications=publications,
                     publication_mode="recovery_with_all_scoring_writers_stopped",
                     recovery_evidence=str(evidence.relative_to(auditor.ROOT)),
                     previous_failed_worker_pid=state["pid"])
        state.pop("error", None)
        auditor.write_json(tail.STATUS, state)
        result = {"state": "passed", "checked_at": auditor.now(), "api_calls": 0,
                  "publication_mode": state["publication_mode"], "publications": publications,
                  "previous_scheduler_pid": scheduler["pid"], "all_scoring_writers_stopped": True,
                  "independent_local_audit_sha256": auditor.sha256(local_report),
                  "independent_prefix_audit_sha256": auditor.sha256(auditor.RUN / "verification_smfa_retain_500.json"),
                  "tail_status_sha256": auditor.sha256(tail.STATUS),
                  "entrypoint_sha256": auditor.sha256(Path(__file__))}
        auditor.write_json(evidence / "publication.json", result)
        print(json.dumps({"state": "passed", "added_scores": 3800,
                          "previous_scores_preserved": 19000, "api_calls": 0,
                          "evidence": str(evidence)}))


if __name__ == "__main__":
    stamp = auditor.now().replace(":", "").replace(".", "_")
    publish(auditor.RUN / f"smfa_tail_publication_recovery_{stamp}")
