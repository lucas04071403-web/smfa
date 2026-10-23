"""Exercise recovery against real question identities and synthetic judge replies."""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from openai.types.chat import ChatCompletion

import recover_judge_cache as recovery
import score

ROOT = recovery.ROOT
module = importlib.import_module("eval_fact")
checks = []

bad_reply = '{"Factuality Score":8,"Justification":"It' + "\\'" + 's precise."}'
for raw in (bad_reply, "```json\n" + bad_reply + "\n```"):
    normalized, parsed, positions = recovery.normalize(raw, module)
    assert parsed == {"Factuality Score": 8, "Justification": "It's precise."}
    assert len(positions) == 1
    assert module.convert_markdown_json_to_json(normalized) == parsed
checks.append("Bare and fenced apostrophe repairs preserve score and text")

valid = json.dumps({"Factuality Score": 9, "Justification": "An already valid literal \\' is unchanged."})
assert recovery.normalize(valid, module) == (valid, json.loads(valid), [])
checks.append("Already valid backslashes and apostrophes are unchanged")
for raw in (r'{"Factuality Score":8,"Justification":"bad\q"}',
            '{"Factuality Score":11,"Justification":"bad"}',
            '{"Factuality Score":8}',
            '{"Factuality Score":8,"Justification":"unterminated}'):
    try:
        recovery.normalize(raw, module)
    except ValueError:
        pass
    else:
        raise AssertionError("Unexpected repair accepted")
checks.append("Other malformed JSON and invalid scores are refused")
try:
    recovery.stopped({"state": "failed", "pid": os.getpid()})
except ValueError:
    pass
else:
    raise AssertionError("Live process accepted")
checks.append("A live process with a failed state is refused")

source = ROOT / "smfa_run/early_original_retain_625/retain_generation_results.json"
questions = json.loads(source.read_text())["Generation_Questions"][:19]
request_calls = []


class Client:
    def with_options(self, **kwargs):
        assert kwargs == {"timeout": 120, "max_retries": 2}
        return self

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        request_calls.append(kwargs)
        return ChatCompletion(id="offline-fixture", object="chat.completion", created=0,
                              model="offline-fixture", choices=[{
                                  "index": 0, "finish_reason": "stop",
                                  "message": {"role": "assistant", "content": bad_reply},
                              }])


with tempfile.TemporaryDirectory(prefix="smfa_recovery_validation_") as temporary:
    root = Path(temporary)
    recovery.ROOT = root
    recovery.RUN = root / "smfa_run"
    recovery.RUN.mkdir()
    work = recovery.RUN / "early_original_retain_1"
    work.mkdir()
    input_path = work / "retain_generation_results.json"
    input_path.write_text(json.dumps({"Generation_Questions": questions}))
    recovery.auditor.write_json(recovery.RUN / "manifest.json",
                               {"conda_prefix": sys.prefix, "source_sha256": {}})
    state = {"state": "failed", "pid": 99999999, "profiles": 1, "split": "retain",
             "stage": "factuality", "questions": 19,
             "input_sha256": recovery.auditor.sha256(input_path)}
    recovery.auditor.write_json(recovery.RUN / "early_original_retain_1_status.json", state)
    recovery.auditor.write_json(recovery.RUN / "original_judge_queue_status.json",
                               {"state": "failed", "pid": 99999998})
    recovery.auditor.write_json(recovery.RUN / "parallel_original_status.json",
                               {"staging": str(root / "staging")})
    (recovery.RUN / "early_original_retain_1.log").write_text("Offline fixture, no API calls\n")
    records = []
    for index, question in enumerate(questions):
        result = {"Task Type": "Generation", "Question Type": question["question type"],
                  "Image_ID": question.get("image_id"), "image_type": question.get("image_type"),
                  "Question": question["question"], "Generated Answer": question["generated_answer"],
                  "Ground Truth": question["ground_truth"],
                  "Factuality Score": None if index == 0 else 9,
                  "Justification": None if index == 0 else "Offline fixture."}
        records.append({"index": index, "digest": score.digest(question), "result": result})
    cache = work / "retain_generation_results_factuality.jsonl"
    cache.write_text("".join(json.dumps(row) + "\n" for row in records))
    original = cache.read_bytes().splitlines(keepends=True)
    protected_cache = work / "retain_generation_results_meaningful.jsonl"
    protected_cache.write_text("Protected fixture\n")
    module.client = Client()
    generation_checks = []
    recovery.helper.require_generation_running = lambda split, path: generation_checks.append((split, str(path)))
    recovery.recover(argparse.Namespace(profiles=1, metric="factuality", indexes=[0]))
    recovered = cache.read_bytes().splitlines(keepends=True)
    assert original[1:] == recovered[:18]
    assert json.loads(recovered[-1])["result"]["Factuality Score"] == 8
    assert len(request_calls) == 1 and request_calls[0]["model"] == "qwen-plus"
    assert set(request_calls[0]) == {"model", "messages"}
    assert len(generation_checks) == 2
    assert protected_cache.read_text() == "Protected fixture\n"
    evidence = next(recovery.RUN.glob("retain_1_factuality_json_recovery_*"))
    report = json.loads((evidence / "recovery.json").read_text())
    assert report["valid_records_preserved_byte_for_byte"] == 18
    assert report["api_calls_after_normal_retries_failed"] == 1
checks.append("Official processing and validation replace one failed score while preserving 18 valid record bytes")
checks.append("Other cache is unchanged; request uses official model/messages, one synthetic completion, no network")
record = {"state": "passed", "checked_at": recovery.auditor.now(),
          "script_sha256": recovery.auditor.sha256(Path(recovery.__file__)),
          "validator_sha256": recovery.auditor.sha256(Path(__file__)),
          "checks": checks, "real_api_calls": 0, "formal_inputs_modified": False}
recovery.auditor.write_json(ROOT / "smfa_run/recover_judge_cache_validation.json", record)
print(json.dumps(record, indent=2))
