"""Recover the one inspected full SMFA factuality null after the scorer closes."""

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

import audit_final as audit
import early_original_forget as helper
import recover_judge_cache as recovery
import score


def live(pid):
    return audit.live(pid)


def main():
    os.umask(0o077)
    manifest = audit.read_json(audit.RUN / "manifest.json")
    audit.require(sys.prefix == manifest["conda_prefix"], "Use the recorded Conda environment")
    for relative, expected in manifest["source_sha256"].items():
        audit.require(audit.sha256(audit.ROOT / relative) == expected, "Frozen source changed")
    with (audit.RUN / "full_smfa_recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = audit.RUN / "status.json"
        state = audit.read_json(state_path)
        audit.require(state["state"] == "failed" and state["stage"] == "score_SMFA",
                      "Formal SMFA scoring has not closed with the inspected failure")
        audit.require(not live(state["pid"]) and not live(state["child_pid"]),
                      "Formal scoring process is still live")
        folder = audit.EVALUATION / "SMFA"
        source = folder / "retain_generation_results.json"
        questions = audit.read_json(source)["Generation_Questions"]
        audit.require(len(questions) == 18050, "Full retain generation is incomplete")
        cache = folder / "retain_generation_results_factuality.jsonl"
        original = cache.read_bytes()
        lines = original.splitlines(keepends=True)
        audit.require(original.endswith(b"\n") and len(lines) == len(questions), "Cache is not closed")
        rows = [json.loads(line) for line in lines]
        audit.require({row["index"] for row in rows} == set(range(len(questions))),
                      "Cache indexes are incomplete or duplicated")
        invalid = [row["index"] for row in rows if row["result"].get("Factuality Score") is None]
        audit.require(invalid == [11348], "Inspected null-score scope changed")
        valid = helper.verified_cache(cache, questions, "Factuality Score")
        audit.require(set(valid) == set(range(len(questions))) - set(invalid), "Another score is invalid")
        module = importlib.import_module("eval_fact")
        audit.require(module.client is not None, "Judge credential is absent")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        evidence = audit.RUN / f"smfa_full_factuality_recovery_{stamp}"
        evidence.mkdir()
        for path in (state_path, cache, audit.RUN / "score_SMFA.log"):
            shutil.copy2(path, evidence / path.name)
        protected = {path: audit.sha256(path) for path in folder.iterdir() if path.is_file() and path != cache}
        protected[state_path] = audit.sha256(state_path)
        client = module.client.with_options(timeout=120, max_retries=2)
        calls = []
        current_index = invalid[0]

        def create(**arguments):
            completion = client.chat.completions.create(**arguments)
            content = completion.choices[0].message.content
            record = {"question_index": current_index, "arguments": arguments,
                      "response": completion.model_dump(mode="json"), "raw_content": content}
            calls.append(record)
            audit.write_json(evidence / "responses.json", calls)
            normalized, parsed, repairs = recovery.normalize(content, module)
            record.update(normalized_content=normalized, parsed_response=parsed,
                          illegal_apostrophe_escape_positions=repairs)
            audit.write_json(evidence / "responses.json", calls)
            repaired = completion.model_copy(deep=True)
            repaired.choices[0].message.content = normalized
            return repaired

        module.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        output, totals = ({kind: [] for kind in score.KINDS} for _ in range(2))
        module.process_single_question(questions[current_index], output, totals)
        result = output[questions[current_index]["question type"]][0]
        audit.require(module.validate_structure(result)[0]
                      and type(result["Factuality Score"]) is int
                      and result["Justification"].strip(), "Recovered result is invalid")
        captured = calls[-1]["parsed_response"]
        audit.require(all(result[key] == captured[key] for key in ("Factuality Score", "Justification")),
                      "Recovered result differs from official response")
        replacement = {"index": current_index, "digest": score.digest(questions[current_index]), "result": result}
        audit.require(cache.read_bytes() == original, "Cache changed during API recovery")
        for path, expected in protected.items():
            audit.require(audit.sha256(path) == expected, "Protected artifact changed")
        repaired = b"".join(line for row, line in zip(rows, lines) if row["index"] != current_index)
        repaired += (json.dumps(replacement) + "\n").encode()
        temporary = cache.with_suffix(".recover.tmp")
        with temporary.open("xb") as handle:
            handle.write(repaired)
            handle.flush()
            os.fsync(handle.fileno())
        audit.require(len(helper.verified_cache(temporary, questions, "Factuality Score")) == len(questions),
                      "Repaired cache failed complete validation")
        temporary.replace(cache)
        result = {"state": "recovered_closed_full_cache", "checked_at": audit.now(),
                  "invalid_indexes": invalid, "valid_records_preserved_byte_for_byte": len(lines) - 1,
                  "api_calls": len(calls), "recovered_score": result["Factuality Score"],
                  "original_cache_sha256": hashlib.sha256(original).hexdigest(),
                  "recovered_cache_sha256": audit.sha256(cache),
                  "protected_files_sha256": {str(path.relative_to(audit.ROOT)): value for path, value in protected.items()},
                  "source_sha256": manifest["source_sha256"],
                  "recovery_script_sha256": audit.sha256(Path(__file__)),
                  "note": "Only the inspected factuality null was requested after the closed formal scorer; all other records were preserved byte-for-byte."}
        audit.write_json(evidence / "recovery.json", result)
        print(json.dumps({"evidence": str(evidence), **result}, indent=2))


if __name__ == "__main__":
    main()
