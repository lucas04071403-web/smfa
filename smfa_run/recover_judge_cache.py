"""Recover explicitly inspected early Original scores after their worker exits."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "smfa_run"
sys.path.insert(0, str(ROOT / "reproduce_smfa"))
import score
import audit_final as auditor
import early_original_forget as helper

METRICS = {
    "factuality": ("eval_fact", "Factuality Score"),
    "meaningful": ("eval_meaningful", "Meaningful Score"),
}


def stopped(state):
    auditor.require(state["state"] == "failed", "Inspect the failed task before recovery")
    proc = Path(f"/proc/{state['pid']}/stat")
    auditor.require(not proc.exists() or proc.read_text().rsplit(")", 1)[1].split()[0] == "Z",
                    "Refusing to recover a live task")


def normalize(content, module):
    auditor.require(isinstance(content, str), "Judge did not return text")
    match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
    begin, end = match.span(1) if match else (0, len(content))
    body = content[begin:end]
    repairs = []
    while True:
        try:
            parsed = json.loads(body)
            break
        except json.JSONDecodeError as error:
            auditor.require(error.msg == "Invalid \\escape" and body[error.pos:error.pos + 2] == "\\'",
                            "Response requires more than an illegal apostrophe escape repair")
            repairs.append(error.pos)
            body = body[:error.pos] + body[error.pos + 1:]
    auditor.require(module.validate_structure(parsed)[0], "Response fails the official validator")
    normalized = content[:begin] + body + content[end:]
    auditor.require(module.convert_markdown_json_to_json(normalized) == parsed,
                    "Official parser does not reproduce the normalized response")
    return normalized, parsed, repairs


def recover(args):
    os.umask(0o077)
    manifest = auditor.read_json(RUN / "manifest.json")
    auditor.require(sys.prefix == manifest["conda_prefix"], "Activate the recorded Conda environment")
    for relative, expected in manifest["source_sha256"].items():
        auditor.require(auditor.sha256(ROOT / relative) == expected, f"Experiment source changed: {relative}")
    label = f"early_original_retain_{args.profiles}"
    state_path = RUN / f"{label}_status.json"
    queue_path = RUN / "original_judge_queue_status.json"
    with (RUN / "early_original.lock").open("a") as worker_lock, \
            (RUN / "original_judge_queue.lock").open("a") as queue_lock:
        for lock in (worker_lock, queue_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, queue = auditor.read_json(state_path), auditor.read_json(queue_path)
        stopped(state)
        stopped(queue)
        auditor.require(state["profiles"] == args.profiles and state["split"] == "retain",
                        "Failed state is for a different input")
        auditor.require(state["stage"] == args.metric, "Failed metric differs from inspected state")
        controller = auditor.read_json(RUN / "parallel_original_status.json")
        helper.require_generation_running("retain", Path(controller["staging"]))
        module_name, field = METRICS[args.metric]
        module = importlib.import_module(module_name)
        auditor.require(module.client is not None, "Judge credential is absent")
        work = RUN / label
        source = work / "retain_generation_results.json"
        auditor.require(auditor.sha256(source) == state["input_sha256"], "Judged input changed")
        questions = auditor.read_json(source)["Generation_Questions"]
        auditor.require(len(questions) == args.profiles * 19 == state["questions"], "Input coverage differs")
        cache = work / f"retain_generation_results_{args.metric}.jsonl"
        original = cache.read_bytes()
        lines = original.splitlines(keepends=True)
        auditor.require(original.endswith(b"\n") and len(lines) == len(questions), "Unexpected cache coverage")
        rows = [json.loads(line) for line in lines]
        auditor.require(all(type(row["index"]) is int for row in rows), "Invalid cache index type")
        auditor.require({row["index"] for row in rows} == set(range(len(questions))),
                        "Duplicate or missing cache indexes")
        invalid = [row["index"] for row in rows if row["result"].get(field) is None]
        auditor.require(set(invalid) == set(args.indexes) and len(args.indexes) == len(invalid),
                        "Missing scores differ from the explicitly inspected indexes")
        verified = helper.verified_cache(cache, questions, field)
        auditor.require(set(verified) == set(range(len(questions))) - set(invalid),
                        "An additional cached score is invalid")
        for row in rows:
            auditor.require(row["digest"] == score.digest(questions[row["index"]]), "Cache digest differs")
            if row["index"] not in invalid:
                auditor.require(module.validate_structure(row["result"])[0], "Cached result fails validator")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        evidence = RUN / f"retain_{args.profiles}_{args.metric}_json_recovery_{stamp}"
        evidence.mkdir()
        for path in (state_path, queue_path, cache, RUN / f"{label}.log"):
            shutil.copy2(path, evidence / path.name)
        protected = {path: auditor.sha256(path) for path in work.iterdir() if path.is_file() and path != cache}
        protected.update({state_path: auditor.sha256(state_path), queue_path: auditor.sha256(queue_path)})
        real_client = module.client.with_options(timeout=120, max_retries=2)
        calls = []

        def create(**arguments):
            completion = real_client.chat.completions.create(**arguments)
            content = completion.choices[0].message.content
            record = {"question_index": current_index, "arguments": arguments,
                      "response": completion.model_dump(mode="json"), "raw_content": content}
            calls.append(record)
            auditor.write_json(evidence / "responses.json", calls)
            normalized, parsed, repairs = normalize(content, module)
            record.update(normalized_content=normalized, parsed_response=parsed,
                          illegal_apostrophe_escape_positions=repairs)
            auditor.write_json(evidence / "responses.json", calls)
            repaired = completion.model_copy(deep=True)
            repaired.choices[0].message.content = normalized
            return repaired

        module.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        replacements = []
        for current_index in sorted(invalid):
            output = {kind: [] for kind in score.KINDS}
            totals = {kind: [] for kind in score.KINDS}
            question = questions[current_index]
            module.process_single_question(question, output, totals)
            result = output[question["question type"]][0]
            auditor.require(module.validate_structure(result)[0], "Recovery failed the official validator")
            auditor.require(type(result[field]) is int and result["Justification"].strip(), "Invalid recovered score")
            captured = [call["parsed_response"] for call in calls
                        if call["question_index"] == current_index and "parsed_response" in call]
            auditor.require(bool(captured) and all(result[key] == captured[-1][key]
                                                 for key in (field, "Justification")),
                            "Recovered score differs from the captured response")
            replacements.append({"index": current_index, "digest": score.digest(question), "result": result})
        stopped(auditor.read_json(state_path))
        stopped(auditor.read_json(queue_path))
        helper.require_generation_running("retain", Path(controller["staging"]))
        auditor.require(cache.read_bytes() == original, "Failed cache changed during recovery")
        for path, expected in protected.items():
            auditor.require(auditor.sha256(path) == expected, f"Another task file changed: {path.name}")
        valid_lines = [line for line, row in zip(lines, rows) if row["index"] not in invalid]
        recovered = b"".join(valid_lines + [(json.dumps(row) + "\n").encode() for row in replacements])
        before = {row["index"]: line for row, line in zip(rows, lines) if row["index"] not in invalid}
        after = {json.loads(line)["index"]: line for line in recovered.splitlines(keepends=True)}
        auditor.require(len(after) == len(questions) and all(after[index] == line for index, line in before.items()),
                        "Recovery changed an existing valid record")
        temporary = cache.with_suffix(".recover.tmp")
        with temporary.open("xb") as handle:
            handle.write(recovered)
            handle.flush()
            os.fsync(handle.fileno())
        auditor.require(len(helper.verified_cache(temporary, questions, field)) == len(questions),
                        "Recovered cache failed coverage or identity checks")
        temporary.replace(cache)
        recovery = {"state": "recovered_local_cache", "checked_at": auditor.now(),
                    "profiles": args.profiles, "metric": args.metric, "invalid_indexes": sorted(invalid),
                    "valid_records_preserved_byte_for_byte": len(valid_lines),
                    "api_calls_after_normal_retries_failed": len(calls),
                    "original_cache_sha256": hashlib.sha256(original).hexdigest(),
                    "recovered_cache_sha256": auditor.sha256(cache),
                    "other_task_files_unchanged": {str(path.relative_to(ROOT)): value
                                                   for path, value in protected.items()},
                    "recovered_scores": {str(row["index"]): row["result"][field] for row in replacements},
                    "source_sha256": manifest["source_sha256"],
                    "recovery_script_sha256": auditor.sha256(Path(__file__)),
                    "note": "Only illegal JSON apostrophe escapes were normalized. Official messages, model arguments and validator were retained. The original helper must still verify and publish this prefix."}
        auditor.write_json(evidence / "recovery.json", recovery)
        print(json.dumps({"evidence": str(evidence.relative_to(ROOT)), **recovery}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=int, required=True)
    parser.add_argument("--metric", choices=tuple(METRICS), required=True)
    parser.add_argument("--indexes", nargs="+", type=int, required=True)
    recover(parser.parse_args())
