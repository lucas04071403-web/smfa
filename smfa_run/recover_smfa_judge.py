"""Recover one explicitly inspected SMFA judge record after its worker exits."""

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
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "smfa_run"
REPO = ROOT / "S-MLLMUn"
sys.path.insert(0, str(ROOT / "reproduce_smfa"))
import score
import audit_final as auditor
import early_original_forget as helper


def live(pid):
    path = Path(f"/proc/{pid}/stat")
    try:
        return path.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def normalize(content, module):
    auditor.require(isinstance(content, str), "Judge did not return text")
    match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
    begin, end = match.span(1) if match else (0, len(content))
    body, repairs = content[begin:end], []
    while True:
        try:
            parsed = json.loads(body)
            break
        except json.JSONDecodeError as error:
            auditor.require(error.msg == "Invalid \\escape" and body[error.pos:error.pos + 2] == "\\'",
                            "Response requires more than illegal apostrophe escape repair")
            repairs.append(error.pos)
            body = body[:error.pos] + body[error.pos + 1:]
    auditor.require(module.validate_structure(parsed)[0], "Response fails the official validator")
    normalized = content[:begin] + body + content[end:]
    auditor.require(module.convert_markdown_json_to_json(normalized) == parsed,
                    "Official parser does not reproduce normalized response")
    return normalized, parsed, repairs


def main(args):
    os.umask(0o077)
    manifest = auditor.read_json(RUN / "manifest.json")
    auditor.require(sys.prefix == manifest["conda_prefix"], "Use the recorded Conda environment")
    for relative, expected in manifest["source_sha256"].items():
        auditor.require(auditor.sha256(ROOT / relative) == expected, f"Frozen source changed: {relative}")
    name = f"early_smfa_{args.split}_{args.profiles}"
    state_path, queue_path = RUN / f"{name}_status.json", RUN / "early_smfa_queue_status.json"
    with (RUN / "early_smfa.lock").open("a") as worker_lock, (RUN / "early_smfa_queue.lock").open("a") as queue_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, queue = auditor.read_json(state_path), auditor.read_json(queue_path)
        auditor.require(state["state"] == "failed", "Prefix worker has not failed")
        auditor.require(queue["state"] == "failed", "Queue has not stopped after the failed worker")
        auditor.require(not live(state["pid"]) and not live(queue["pid"]), "Refusing to recover live processes")
        auditor.require(state["stage"] == args.metric and state["split"] == args.split,
                        "Failed prefix does not match requested recovery")
        folder = RUN / name
        source = folder / f"{args.split}_generation_results.json"
        questions = auditor.read_json(source)["Generation_Questions"]
        auditor.require(len(questions) == args.profiles * 19 == state["questions"], "Question coverage differs")
        metric_module, field = importlib.import_module("eval_fact"), "Factuality Score"
        if args.metric == "meaningful":
            metric_module, field = importlib.import_module("eval_meaningful"), "Meaningful Score"
        cache = folder / f"{args.split}_generation_results_{args.metric}.jsonl"
        original = cache.read_bytes()
        lines = original.splitlines(keepends=True)
        auditor.require(original.endswith(b"\n") and len(lines) == len(questions), "Unexpected cache coverage")
        rows = [json.loads(line) for line in lines]
        invalid = [row["index"] for row in rows if row["result"].get(field) is None]
        auditor.require(set(invalid) == set(args.indexes), "Invalid indexes differ from explicit inspection")
        verified = helper.verified_cache(cache, questions, field)
        auditor.require(set(verified) == set(range(len(questions))) - set(invalid), "Other cached scores are invalid")
        for row in rows:
            auditor.require(row["digest"] == score.digest(questions[row["index"]]), "Cache digest differs")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        evidence = RUN / f"smfa_{args.split}_{args.profiles}_{args.metric}_json_recovery_{stamp}"
        evidence.mkdir()
        for path in (state_path, queue_path, cache, RUN / f"{name}.log"):
            shutil.copy2(path, evidence / path.name)
        protected = {path: auditor.sha256(path) for path in folder.iterdir() if path.is_file() and path != cache}
        protected.update({state_path: auditor.sha256(state_path), queue_path: auditor.sha256(queue_path)})
        real_client = metric_module.client.with_options(timeout=120, max_retries=2)
        calls, current_index = [], None

        def create(**arguments):
            completion = real_client.chat.completions.create(**arguments)
            content = completion.choices[0].message.content
            record = {"question_index": current_index, "arguments": arguments,
                      "response": completion.model_dump(mode="json"), "raw_content": content}
            calls.append(record)
            auditor.write_json(evidence / "responses.json", calls)
            normalized, parsed, repairs = normalize(content, metric_module)
            record.update(normalized_content=normalized, parsed_response=parsed,
                          illegal_apostrophe_escape_positions=repairs)
            auditor.write_json(evidence / "responses.json", calls)
            repaired = completion.model_copy(deep=True)
            repaired.choices[0].message.content = normalized
            return repaired

        metric_module.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        replacements = []
        for current_index in sorted(invalid):
            output = {kind: [] for kind in score.KINDS}
            totals = {kind: [] for kind in score.KINDS}
            question = questions[current_index]
            metric_module.process_single_question(question, output, totals)
            result = output[question["question type"]][0]
            auditor.require(metric_module.validate_structure(result)[0], "Recovered response fails validator")
            auditor.require(type(result[field]) is int and 0 <= result[field] <= 10, "Recovered score is invalid")
            replacements.append({"index": current_index, "digest": score.digest(question), "result": result})
        auditor.require(cache.read_bytes() == original, "Failed cache changed during recovery")
        for path, expected in protected.items():
            auditor.require(auditor.sha256(path) == expected, f"Protected file changed: {path.name}")
        valid_lines = [line for line, row in zip(lines, rows) if row["index"] not in invalid]
        recovered = b"".join(valid_lines + [(json.dumps(row) + "\n").encode() for row in replacements])
        temporary = cache.with_suffix(".recover.tmp")
        with temporary.open("xb") as handle:
            handle.write(recovered)
            handle.flush()
            os.fsync(handle.fileno())
        auditor.require(len(helper.verified_cache(temporary, questions, field)) == len(questions),
                        "Recovered cache failed independent coverage")
        temporary.replace(cache)
        recovery = {"state": "recovered_local_cache", "checked_at": auditor.now(), "split": args.split,
                    "profiles": args.profiles, "metric": args.metric, "invalid_indexes": sorted(invalid),
                    "valid_records_preserved_byte_for_byte": len(valid_lines), "api_calls": len(calls),
                    "original_cache_sha256": hashlib.sha256(original).hexdigest(),
                    "recovered_cache_sha256": auditor.sha256(cache),
                    "recovered_scores": {str(row["index"]): row["result"][field] for row in replacements},
                    "source_sha256": manifest["source_sha256"], "recovery_script_sha256": auditor.sha256(Path(__file__)),
                    "note": "Only illegal JSON apostrophe escapes were normalized; official prompt, model and validator were retained."}
        auditor.write_json(evidence / "recovery.json", recovery)
        print(json.dumps({"evidence": str(evidence.relative_to(ROOT)), **recovery}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("forget", "retain"), required=True)
    parser.add_argument("--profiles", type=int, required=True)
    parser.add_argument("--metric", choices=("factuality", "meaningful"), required=True)
    parser.add_argument("--indexes", nargs="+", type=int, required=True)
    main(parser.parse_args())
