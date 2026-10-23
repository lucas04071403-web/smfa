"""Recover the two failed retain-500 responses with full provenance."""

from datetime import datetime, timezone
import hashlib
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
import eval_meaningful as official


def stopped(state):
    auditor.require(state["state"] == "failed", "The failed task must be inspected before recovery")
    proc = Path(f"/proc/{state['pid']}/stat")
    auditor.require(not proc.exists() or proc.read_text().rsplit(")", 1)[1].split()[0] == "Z",
                    "Refusing to recover a live task")


def normalize(content):
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
    normalized = content[:begin] + body + content[end:]
    auditor.require(official.validate_structure(parsed)[0], "Response fails the official validator")
    return normalized, parsed, repairs


def main():
    os.umask(0o077)
    manifest = auditor.read_json(RUN / "manifest.json")
    auditor.require(sys.prefix == manifest["conda_prefix"], "Activate the recorded Conda environment")
    for relative, expected in manifest["source_sha256"].items():
        auditor.require(auditor.sha256(ROOT / relative) == expected, f"Experiment source changed: {relative}")
    state_path = RUN / "early_original_retain_500_status.json"
    queue_path = RUN / "original_judge_queue_status.json"
    state, queue = auditor.read_json(state_path), auditor.read_json(queue_path)
    stopped(state)
    stopped(queue)
    auditor.require(official.client is not None, "Judge credential is absent")
    work = RUN / "early_original_retain_500"
    source = work / "retain_generation_results.json"
    auditor.require(auditor.sha256(source) == state["input_sha256"], "Judged input changed")
    questions = auditor.read_json(source)["Generation_Questions"]
    cache = work / "retain_generation_results_meaningful.jsonl"
    original = cache.read_bytes()
    lines = original.splitlines(keepends=True)
    auditor.require(original.endswith(b"\n") and len(lines) == len(questions) == 9500,
                    "Unexpected failed cache coverage")
    rows = [json.loads(line) for line in lines]
    auditor.require({row["index"] for row in rows} == set(range(9500)), "Duplicate or missing cache indexes")
    invalid = [row["index"] for row in rows if row["result"].get("Meaningful Score") is None]
    auditor.require(set(invalid) == {9397, 9404}, "Failed questions differ from the inspected cache")
    for row in rows:
        auditor.require(row["digest"] == score.digest(questions[row["index"]]), "Cache digest differs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = RUN / f"retain_500_json_recovery_{stamp}"
    evidence.mkdir()
    for path in (state_path, queue_path, cache, RUN / "early_original_retain_500.log"):
        shutil.copy2(path, evidence / path.name)
    factuality_path = work / "retain_generation_results_factuality.jsonl"
    factuality_hash = auditor.sha256(factuality_path)
    real_client = official.client.with_options(timeout=120, max_retries=2)
    calls = []

    def create(**arguments):
        completion = real_client.chat.completions.create(**arguments)
        content = completion.choices[0].message.content
        record = {"question_index": current_index, "arguments": arguments,
                  "response": completion.model_dump(mode="json"), "raw_content": content}
        calls.append(record)
        auditor.write_json(evidence / "responses.json", calls)
        normalized, parsed, repairs = normalize(content)
        record.update(normalized_content=normalized, parsed_response=parsed,
                      illegal_apostrophe_escape_positions=repairs)
        auditor.write_json(evidence / "responses.json", calls)
        repaired = completion.model_copy(deep=True)
        repaired.choices[0].message.content = normalized
        return repaired

    official.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    replacements = []
    for current_index in sorted(invalid):
        output = {kind: [] for kind in score.KINDS}
        totals = {kind: [] for kind in score.KINDS}
        question = questions[current_index]
        official.process_single_question(question, output, totals)
        result = output[question["question type"]][0]
        auditor.require(official.validate_structure(result)[0], "Recovery did not yield a valid score")
        auditor.require(type(result["Meaningful Score"]) is int and result["Justification"].strip(),
                        "Recovery score or justification is invalid")
        replacements.append({"index": current_index, "digest": score.digest(question), "result": result})
    stopped(auditor.read_json(state_path))
    stopped(auditor.read_json(queue_path))
    auditor.require(cache.read_bytes() == original, "Failed cache changed during recovery")
    valid_lines = [line for line, row in zip(lines, rows) if row["index"] not in invalid]
    replacement_lines = [(json.dumps(row) + "\n").encode() for row in replacements]
    recovered = b"".join(valid_lines + replacement_lines)
    auditor.require(len(valid_lines) == 9498 and len(recovered.splitlines()) == 9500,
                    "Unexpected recovery record count")
    before = {row["index"]: line for row, line in zip(rows, lines) if row["index"] not in invalid}
    after = {json.loads(line)["index"]: line for line in recovered.splitlines(keepends=True)}
    auditor.require(all(after[index] == line for index, line in before.items()), "A valid record changed")
    temporary = cache.with_suffix(".recover.tmp")
    with temporary.open("xb") as handle:
        handle.write(recovered)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(cache)
    auditor.require(auditor.sha256(factuality_path) == factuality_hash, "Factuality cache changed")
    recovery = {"state": "recovered_local_cache", "checked_at": auditor.now(),
                "profiles": 500, "invalid_indexes": sorted(invalid),
                "valid_records_preserved_byte_for_byte": len(valid_lines),
                "api_calls_after_normal_retries_failed": len(calls),
                "original_cache_sha256": hashlib.sha256(original).hexdigest(),
                "recovered_cache_sha256": auditor.sha256(cache),
                "factuality_cache_unchanged_sha256": factuality_hash,
                "recovered_scores": {str(row["index"]): row["result"]["Meaningful Score"] for row in replacements},
                "source_sha256": manifest["source_sha256"], "recovery_script_sha256": auditor.sha256(Path(__file__)),
                "note": "Only illegal JSON apostrophe escapes were repaired. Official messages, model arguments and validator were retained. The original helper must still verify and publish this prefix."}
    auditor.write_json(evidence / "recovery.json", recovery)
    print(json.dumps({"evidence": str(evidence.relative_to(ROOT)), **recovery}, indent=2))


if __name__ == "__main__":
    main()
