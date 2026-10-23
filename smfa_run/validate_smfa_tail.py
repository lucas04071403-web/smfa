"""Validate tail index mapping, publication and scheduler recovery offline."""

from collections import Counter
import json
from pathlib import Path
import shutil
import signal
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import score_smfa_tail as tail


def no_api(*args, **kwargs):
    raise AssertionError("Offline validation attempted an API request")


def main():
    import pandas as pd

    root, real_dest = tail.ROOT, tail.DEST
    source = tail.WORK / "retain_generation_results.json"
    questions = tail.auditor.read_json(source)["Generation_Questions"]
    checks = []
    with tempfile.TemporaryDirectory(prefix="smfa_tail_validation_") as temporary:
        base = Path(temporary)
        work, dest, run = base / "tail", base / "destination", base / "run"
        for directory in (work, dest, run):
            directory.mkdir()
        shutil.copy2(source, work / source.name)
        for index in [0, *range(tail.START, tail.STOP)]:
            relative = Path(".profiles/retain") / f"{index:04d}" / source.name
            (dest / relative).parent.mkdir(parents=True)
            shutil.copy2(real_dest / relative, dest / relative)
        initial = {}
        for _, metric, field in tail.METRICS:
            name = f"retain_generation_results_{metric}.jsonl"
            shared = {}
            for line in (real_dest / name).read_bytes().splitlines(keepends=True):
                item = json.loads(line)
                if item["index"] < 19:
                    shared[item["index"]] = line
            initial[metric] = b"".join(shared[i] for i in range(19))
            (dest / name).write_bytes(initial[metric])
            local = []
            for index, question in enumerate(questions):
                result = {"Task Type": "Generation", "Question Type": question["question type"],
                          "Image_ID": question.get("image_id"), "image_type": question.get("image_type"),
                          "Question": question["question"], "Generated Answer": question["generated_answer"],
                          "Ground Truth": question["ground_truth"], field: 7,
                          "Justification": "Offline fixture; no API call."}
                local.append({"index": index, "digest": tail.score.digest(question), "result": result})
            cache = work / name
            cache.write_text("".join(json.dumps(item) + "\n" for item in local))
            before = cache.read_bytes()
            tail.score.score_file(SimpleNamespace(process_single_question=no_api), work / source.name,
                                  field, metric, 1)
            assert cache.read_bytes() == before
        rows = pd.read_parquet(root / "S-MLLMUn/data/S-MLLMUn-data/eval_data/retain_5/train-00000-of-00001.parquet")
        measured = tail.auditor.audit_split(work, "retain", rows.iloc[tail.START:tail.STOP])
        assert measured["questions"] == 1900
        checks.append("Both complete metric caches rebuild summaries and pass the split audit without API calls")

        scheduler, worker = 99999981, 99999982
        status = {"state": "running", "stage": "judging", "pid": scheduler,
                  "child_pid": worker, "target": ["retain", 500]}
        queue_path = run / "queue.json"
        tail.auditor.write_json(queue_path, status)
        worker_status = run / "early_smfa_retain_500_status.json"
        tail.auditor.write_json(worker_status, {"state": "complete", "stage": "meaningful", "published": True})
        prefix_work = run / "early_smfa_retain_500"
        prefix_work.mkdir()
        (prefix_work / "retain_generation_results_meaningful.jsonl").write_bytes(b"{}\n" * 9500)
        signals = []
        guards = []
        scheduler_state = ["S"]

        def fake_process(pid):
            assert pid == scheduler
            return scheduler_state[0], "12345"

        def fake_kill(pid, value):
            assert pid == scheduler
            signals.append(value)
            scheduler_state[0] = "T" if value == signal.SIGSTOP else "S"

        def fake_guard(*args, **kwargs):
            counts = Counter()
            guards.append(counts)
            return SimpleNamespace(terminate=lambda: counts.update(["terminate"]),
                                   wait=lambda: counts.update(["wait"]))

        with patch.multiple(tail, RUN=run, WORK=work, DEST=dest, STATUS=run / "tail_status.json"), \
                patch.object(tail.queue, "STATUS", queue_path), \
                patch.object(tail, "frozen", return_value={}), \
                patch.object(tail, "snapshot", return_value=(questions, {})), \
                patch.object(tail, "generation_available", return_value=True), \
                patch.object(tail, "process", side_effect=fake_process), \
                patch.object(tail.queue, "live", return_value=False), \
                patch.object(tail.os, "kill", side_effect=fake_kill), \
                patch.object(tail.subprocess, "Popen", side_effect=fake_guard):
            state = {"state": "running"}
            tail.publish(state, questions)
            assert state["state"] == "complete" and state["published"]
            assert signals == [signal.SIGSTOP, signal.SIGCONT]
            assert guards == [Counter(terminate=1, wait=1)]
            for _, metric, field in tail.METRICS:
                path = dest / f"retain_generation_results_{metric}.jsonl"
                data, actual = tail.valid_shared(path, field)
                assert data.startswith(initial[metric]) and len(actual) == 1919
                mapped = tail.remap(work / path.name, questions, field)
                assert set(mapped) == set(range(16150, 18050))
                assert all(actual[i] == item for i, item in mapped.items())
                prefix = dest / ".profiles/retain/0000" / source.name
                prefix_questions = tail.auditor.read_json(prefix)["Generation_Questions"]
                prefix_cache = base / f"prefix_{metric}.jsonl"
                prefix_cache.write_bytes(initial[metric])
                assert tail.helper.append_verified_cache(prefix_cache, path, prefix_questions, field) == 0
                assert path.read_bytes() == data
            checks.append("All 1,900 local indexes map to 16,150 through 18,049; both publications preserve existing bytes")
            checks.append("The unchanged prefix publication helper preserves noncontiguous tail scores")

            signals.clear()
            for _, metric, _ in tail.METRICS:
                (dest / f"retain_generation_results_{metric}.jsonl").write_bytes(initial[metric])
            with patch.object(tail, "remap", side_effect=RuntimeError("Injected validation failure")):
                try:
                    tail.publish({"state": "running"}, questions)
                except RuntimeError as error:
                    assert str(error) == "Injected validation failure"
                else:
                    raise AssertionError("Injected failure was swallowed")
            assert signals == [signal.SIGSTOP, signal.SIGCONT]
            assert all(g == Counter(terminate=1, wait=1) for g in guards)
            for _, metric, _ in tail.METRICS:
                assert (dest / f"retain_generation_results_{metric}.jsonl").read_bytes() == initial[metric]
            checks.append("Publication failure resumes the scheduler and closes its guard without modifying either shared cache")

        guard_signals = []
        with patch.object(tail, "process", side_effect=[None, ("T", "12345")]), \
                patch.object(tail.os, "kill", side_effect=lambda *args: guard_signals.append(args)):
            tail.resume_guard([str(scheduler), "12345", "99999983"])
        assert guard_signals == [(scheduler, signal.SIGCONT)]
        checks.append("A dead publication parent causes the independent guard to resume the matching scheduler")
    result = {"state": "passed", "checked_at": tail.auditor.now(), "checks": checks,
              "api_calls": 0, "real_process_signals": 0, "formal_inputs_modified": False,
              "source_sha256": tail.auditor.sha256(Path(tail.__file__)),
              "validation_sha256": tail.auditor.sha256(Path(__file__))}
    tail.auditor.write_json(root / "smfa_run/smfa_tail_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
