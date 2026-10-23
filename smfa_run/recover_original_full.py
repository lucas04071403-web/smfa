"""Repair inspected missing Original retain scores only after the full worker exits."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import audit_final as auditor
import early_original_forget as helper
import recover_judge_cache as prior_recovery

ROOT, RUN = auditor.ROOT, auditor.RUN
score = prior_recovery.score
FIELD = "Meaningful Score"


def require_closed(state):
    auditor.require(state["state"] == "failed" and state["stage"] == "score_Original",
                    "Full Original scoring must have failed before recovery")
    auditor.require(state["completed_stages"] == ["evaluate_Original"],
                    "Original generation has not completed independently")
    for name in ("pid", "child_pid"):
        auditor.require(name in state and not auditor.live(state[name]),
                        f"Refusing recovery while the Original {name} is live")


def inspect_cache(cache, questions, indexes, module):
    original = cache.read_bytes()
    lines = original.splitlines(keepends=True)
    auditor.require(original.endswith(b"\n") and len(lines) == len(questions),
                    "The closed full cache must contain one record for every question")
    rows = [json.loads(line) for line in lines]
    ids = [row["index"] for row in rows]
    auditor.require(all(type(i) is int for i in ids)
                    and len(set(ids)) == len(questions) and set(ids) == set(range(len(questions))),
                    "The closed cache has duplicate, missing or unexpected indexes")
    invalid = [row["index"] for row in rows if row["result"].get(FIELD) is None]
    auditor.require(len(indexes) == len(set(indexes)) and set(invalid) == set(indexes) and bool(invalid),
                    "Missing scores differ from the explicitly inspected indexes")
    valid = helper.verified_cache(cache, questions, FIELD)
    auditor.require(set(valid) == set(ids) - set(invalid), "An additional score is invalid")
    for row in rows:
        auditor.require(row["digest"] == score.digest(questions[row["index"]]), "Question digest differs")
        if row["index"] not in invalid:
            auditor.require(module.validate_structure(row["result"])[0]
                            and bool(row["result"]["Justification"].strip()),
                            "An existing score fails the official validator")
    return original, lines, rows, sorted(invalid)


def publish_cache(cache, original, lines, rows, replacements, questions):
    replacement_ids = {row["index"] for row in replacements}
    auditor.require(len(replacement_ids) == len(replacements)
                    and replacement_ids == {row["index"] for row in rows if row["result"].get(FIELD) is None},
                    "Replacement coverage differs from missing scores")
    valid = {row["index"]: line for row, line in zip(rows, lines) if row["index"] not in replacement_ids}
    repaired = b"".join([line for row, line in zip(rows, lines) if row["index"] in valid]
                        + [(json.dumps(row) + "\n").encode() for row in replacements])
    after = {json.loads(line)["index"]: line for line in repaired.splitlines(keepends=True)}
    auditor.require(len(after) == len(questions) and all(after[i] == line for i, line in valid.items()),
                    "A valid cached record changed")
    auditor.require(cache.read_bytes() == original, "Closed cache changed before replacement")
    temporary = cache.with_suffix(".recovery.tmp")
    with temporary.open("xb") as handle:
        handle.write(repaired)
        handle.flush()
        os.fsync(handle.fileno())
    auditor.require(len(helper.verified_cache(temporary, questions, FIELD)) == len(questions),
                    "Replacement cache fails complete score and identity checks")
    auditor.require(cache.read_bytes() == original, "Closed cache changed during validation")
    temporary.replace(cache)
    return len(valid)


def recover(indexes):
    os.umask(0o077)
    manifest = auditor.read_json(RUN / "manifest.json")
    auditor.require(sys.prefix == manifest["conda_prefix"], "Activate the recorded Conda environment")
    for relative, expected in manifest["source_sha256"].items():
        auditor.require(auditor.sha256(ROOT / relative) == expected, f"Formal source changed: {relative}")
    status_path = RUN / "parallel_original_status.json"
    with (RUN / "parallel_original.lock").open("a") as controller_lock, \
            (RUN / "early_original.lock").open("a") as early_lock:
        for lock in (controller_lock, early_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = auditor.read_json(status_path)
        require_closed(state)
        pipeline = auditor.read_json(RUN / "status.json")
        auditor.require(pipeline["stage"] not in ("evaluate_Original", "score_Original"),
                        "The formal pipeline may be using Original caches")
        folder = Path(state["staging"])
        auditor.require(folder == auditor.EVALUATION / ".parallel_original/Original"
                        and not Path(state["destination"]).exists(), "Unexpected Original result location")
        generation = auditor.read_json(RUN / "original_generation_audit.json")
        auditor.require(generation["state"] == "passed" and generation["generated_answers_verified"] == 19000,
                        "Complete generation audit is absent")
        source = folder / "retain_generation_results.json"
        auditor.require(auditor.sha256(source) == generation["splits"]["retain"]["generation_sha256"],
                        "Audited Original generation changed")
        questions = auditor.read_json(source)["Generation_Questions"]
        auditor.require(len(questions) == 18050, "Full retain question coverage differs")
        module = importlib.import_module("eval_meaningful")
        auditor.require(module.client is not None, "Activate Conda to load the judge credential")
        cache = folder / "retain_generation_results_meaningful.jsonl"
        original, lines, rows, invalid = inspect_cache(cache, questions, indexes, module)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        evidence = RUN / f"original_full_meaningful_json_recovery_{stamp}"
        evidence.mkdir()
        for path in (cache, status_path, RUN / "parallel_score_Original.log", RUN / "parallel_original.log"):
            shutil.copy2(path, evidence / path.name)
        protected = {path: auditor.sha256(path) for path in folder.iterdir() if path.is_file() and path != cache}
        protected[status_path] = auditor.sha256(status_path)
        calls, replacements = [], []
        real_client = module.client.with_options(timeout=120, max_retries=2)
        original_client = module.client
        current_index = None

        def create(**arguments):
            completion = real_client.chat.completions.create(**arguments)
            content = completion.choices[0].message.content
            record = {"question_index": current_index, "arguments": arguments,
                      "response": completion.model_dump(mode="json"), "raw_content": content}
            calls.append(record)
            auditor.write_json(evidence / "responses.json", calls)
            normalized, parsed, repairs = prior_recovery.normalize(content, module)
            record.update(normalized_content=normalized, parsed_response=parsed,
                          illegal_apostrophe_escape_positions=repairs)
            auditor.write_json(evidence / "responses.json", calls)
            repaired = completion.model_copy(deep=True)
            repaired.choices[0].message.content = normalized
            return repaired

        try:
            module.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            for current_index in invalid:
                output, totals = ({kind: [] for kind in score.KINDS} for _ in range(2))
                question = questions[current_index]
                module.process_single_question(question, output, totals)
                result = output[question["question type"]][0]
                auditor.require(module.validate_structure(result)[0] and type(result[FIELD]) is int
                                and bool(result["Justification"].strip()), "Official recovery validation failed")
                captured = [call["parsed_response"] for call in calls
                            if call["question_index"] == current_index and "parsed_response" in call]
                auditor.require(bool(captured) and all(result[key] == captured[-1][key]
                                                     for key in (FIELD, "Justification")),
                                "Recovered score or justification differs from the API response")
                replacements.append({"index": current_index, "digest": score.digest(question), "result": result})
        finally:
            module.client = original_client
        require_closed(auditor.read_json(status_path))
        auditor.require(auditor.read_json(RUN / "status.json")["stage"]
                        not in ("evaluate_Original", "score_Original"), "Formal scoring began during recovery")
        for path, expected in protected.items():
            auditor.require(auditor.sha256(path) == expected, f"Another result changed: {path.name}")
        preserved = publish_cache(cache, original, lines, rows, replacements, questions)
        result = {"state": "recovered_closed_full_cache", "checked_at": auditor.now(),
                  "scope": "Original retain meaningfulness cache; full scorer and final audits must still run",
                  "invalid_indexes": invalid, "valid_records_preserved_byte_for_byte": preserved,
                  "api_calls_after_normal_retries_failed": len(calls),
                  "recovered_scores": {str(row["index"]): row["result"][FIELD] for row in replacements},
                  "original_cache_sha256": hashlib.sha256(original).hexdigest(),
                  "recovered_cache_sha256": auditor.sha256(cache),
                  "protected_files_sha256": {str(path.relative_to(ROOT)): value for path, value in protected.items()},
                  "source_sha256": manifest["source_sha256"],
                  "recovery_script_sha256": auditor.sha256(Path(__file__)),
                  "normalizer_sha256": auditor.sha256(Path(prior_recovery.__file__)),
                  "note": "Only illegal JSON apostrophe escapes may be normalized; official requests and validator are retained."}
        auditor.write_json(evidence / "recovery.json", result)
        print(json.dumps({"evidence": str(evidence.relative_to(ROOT)), **result}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indexes", nargs="+", type=int, required=True)
    recover(parser.parse_args().indexes)
