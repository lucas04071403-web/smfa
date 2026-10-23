"""Offline integration checks; no API requests or formal result mutations."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "smfa_run"
sys.path.insert(0, str(RUN))
spec = importlib.util.spec_from_file_location("early_smfa_queue", RUN / "early_smfa_queue.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
manifest = m.verify_sources()
checks = []
calls = []


class Client:
    def with_options(self, **kwargs):
        return self


def fake_module(field):
    def process(question, buckets, scores):
        calls.append((field, question["question"]))
        result = {"Task Type": "Generation", "Question Type": question["question type"],
                  "Image_ID": question.get("image_id"), "image_type": question.get("image_type"),
                  "Question": question["question"], "Generated Answer": question["generated_answer"],
                  "Ground Truth": question["ground_truth"], field: 7,
                  "Justification": "Offline synthetic scheduling validation."}
        buckets[question["question type"]].append(result)
        scores[question["question type"]].append(7)
    return SimpleNamespace(client=Client(), process_single_question=process)


modules = {"eval_fact": fake_module("Factuality Score"), "eval_meaningful": fake_module("Meaningful Score")}
real_import = m.importlib.import_module


def import_module(name, *args, **kwargs):
    return modules[name] if name in modules else real_import(name, *args, **kwargs)


with TemporaryDirectory(prefix="smfa_queue_validation_") as directory:
    temporary = Path(directory)
    local_run = temporary / "run"
    local_run.mkdir()
    repo = temporary / "repo"
    repo.mkdir()
    folder = temporary / "results"
    folder.mkdir()
    (repo / "data").symlink_to(ROOT / "S-MLLMUn/data", target_is_directory=True)
    original_state = json.loads((RUN / "parallel_original_status.json").read_text())
    original = Path(original_state["staging"])
    if not original.exists():
        original = Path(original_state["destination"])
    (folder / ".profiles").symlink_to(original / ".profiles", target_is_directory=True)
    (folder / "forget_generation_results.json").symlink_to(original / "forget_generation_results.json")
    generating = {"phase": "generating", "split": "retain", "completed_profiles": 800, "total_profiles": 950}
    with patch.object(m, "RUN", local_run), patch.object(m, "REPO", repo), \
            patch.object(m, "FOLDER", folder), patch.object(m, "STATUS", local_run / "queue_status.json"), \
            patch.object(m, "verify_sources", return_value=manifest), \
            patch.object(m.importlib, "import_module", side_effect=import_module):
        args = SimpleNamespace(split="forget", profiles=50, expected_sha="offline")
        with patch.object(m, "generation_progress", return_value=generating):
            m.worker(args)
        state = json.loads((local_run / "early_smfa_forget_50_status.json").read_text())
        assert state["state"] == "complete" and state["published"] and state["questions"] == 950
        assert len(calls) == 1900
        hashes = {metric: m.sha(folder / f"forget_generation_results_{metric}.jsonl")
                  for metric in ("factuality", "meaningful")}
        with patch.object(m, "generation_progress", return_value=generating):
            m.worker(args)
        assert len(calls) == 1900
        assert all(m.sha(folder / f"forget_generation_results_{metric}.jsonl") == value
                   for metric, value in hashes.items())
        checks.extend([
            "Worker validates 950 real Original forget question and image identities using synthetic scores",
            "Both metric caches publish only after independent audit",
            "Repeating a completed worker reuses every score without calls or duplicate publication",
        ])
        for profiles in (125, 250):
            with patch.object(m, "generation_progress", side_effect=[generating, None]):
                m.worker(SimpleNamespace(split="retain", profiles=profiles, expected_sha="offline"))
            state = json.loads((local_run / f"early_smfa_retain_{profiles}_status.json").read_text())
            assert state["state"] == "complete" and not state["published"]
            assert not (folder / "retain_generation_results_factuality.jsonl").exists()
            assert not (folder / "retain_generation_results_meaningful.jsonl").exists()
        checks.append("Late workers keep audited results isolated and never publish after full-scoring handoff")
        for path in local_run.glob("early_smfa_*_status.json"):
            path.unlink()
        seen, active, sleeps = [], [], []

        class Child:
            def __init__(self, command, **kwargs):
                assert not active
                split = command[command.index("--split") + 1]
                profiles = int(command[command.index("--profiles") + 1])
                active.append((split, profiles))
                seen.append((split, profiles))
                self.pid = 999999
                (local_run / f"early_smfa_{split}_{profiles}_status.json").write_text(json.dumps({
                    "state": "complete", "published": True, "completed_metrics": ["factuality", "meaningful"]}))

            def wait(self):
                assert active
                active.clear()
                return 0

        sequence = [{"phase": "waiting_for_smfa_generation"}, generating]

        def progress():
            return sequence.pop(0) if sequence else generating

        with patch.object(m, "generation_progress", side_effect=progress), \
                patch.object(m.subprocess, "Popen", side_effect=Child), \
                patch.object(m.time, "sleep", side_effect=lambda n: sleeps.append(n)):
            m.queue()
        assert seen == list(m.TARGETS) and sleeps == [30] and not active
        state = json.loads((local_run / "queue_status.json").read_text())
        assert state["state"] == "complete" and state["completed_targets"] == [list(t) for t in m.TARGETS]
        checks.append("Queue waits for generation, runs all seven targets in order, and never overlaps workers")
        for path in local_run.glob("early_smfa_*_status.json"):
            path.unlink()
        with patch.object(m, "generation_progress", return_value=None), patch.object(m.subprocess, "Popen") as spawn:
            m.queue()
            assert not spawn.called
        assert json.loads((local_run / "queue_status.json").read_text())["state"] == "handed_off"
        checks.append("Queue launches no worker after full-scoring handoff")

        class FailedChild:
            pid = 999999

            def wait(self):
                return 1

        with patch.object(m, "generation_progress", return_value=generating), \
                patch.object(m.subprocess, "Popen", return_value=FailedChild()):
            try:
                m.queue()
                raise AssertionError("Failure was ignored")
            except RuntimeError as error:
                assert "prefix failed" in str(error)
        assert json.loads((local_run / "queue_status.json").read_text())["state"] == "failed"
        checks.append("Failed child stops the queue and retains its status for inspection")
    with patch.object(m, "RUN", local_run), patch.object(m, "FOLDER", folder), patch.object(m, "live", return_value=True):
        def status(stage, **extra):
            (local_run / "status.json").write_text(json.dumps(dict(state="running", stage=stage, pid=1, **extra)))

        status("train")
        assert m.generation_progress()["phase"] == "waiting_for_smfa_generation"
        status("evaluate_SMFA")
        assert m.generation_progress()["phase"] == "waiting_for_generator_stage_update"
        status("score_SMFA")
        assert m.generation_progress() is None
        status("evaluate_SMFA", child_pid=2)
        (local_run / "parallel_original_status.json").write_text(json.dumps({"state": "running"}))
        assert m.generation_progress()["phase"] == "waiting_for_original_judging"
        (local_run / "parallel_original_status.json").write_text(json.dumps({"state": "complete"}))
        (folder / "progress.json").write_text("{")
        assert m.generation_progress()["phase"] == "waiting_for_profile_progress"
        (folder / "progress.json").write_text(json.dumps({"split": "retain", "completed_profiles": 948, "total_profiles": 950}))
        assert m.generation_progress() is None
        checks.append("Gates tolerate stage transitions and partial JSON, await Original judging, and stop before the final two profiles")

print(json.dumps({"checked_at": m.runner.now(), "state": "offline_validation_passed",
                  "launcher_sha256": m.sha(Path(m.__file__)), "checks_passed": checks,
                  "synthetic_score_count": len(calls), "real_api_calls": 0,
                  "formal_experiment_files_modified": False}, indent=2))
