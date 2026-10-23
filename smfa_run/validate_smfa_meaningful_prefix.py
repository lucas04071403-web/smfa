"""Exercise isolated metric publication with closed fixtures and no API calls."""

import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import score_smfa_meaningful_prefix as worker


def main():
    questions = worker.audit.read_json(worker.DEST / "retain_generation_results.json")["Generation_Questions"][:38]
    real_records = worker.prefix.records(worker.DEST / "retain_generation_results_meaningful.jsonl")
    checks = []
    with tempfile.TemporaryDirectory(prefix="smfa_meaningful_validation_") as directory:
        base = Path(directory)
        work, dest = base / "work", base / "destination"
        work.mkdir()
        dest.mkdir()
        name = "retain_generation_results_meaningful.jsonl"
        cache, shared = work / name, dest / name
        for index in range(2):
            source = worker.DEST / ".profiles/retain" / f"{index:04d}" / "retain_generation_results.json"
            target = dest / source.relative_to(worker.DEST)
            target.parent.mkdir(parents=True)
            shutil.copy2(source, target)
        before = b"".join(real_records[i] for i in range(19))
        complete = b"".join(real_records[i] for i in range(38))
        with patch.object(worker, "WORK", work), patch.object(worker, "DEST", dest), \
                patch.object(worker.tail, "DEST", dest), patch.object(worker, "guard", return_value={}):
            def reset(data=complete):
                cache.write_bytes(data)
                shared.write_bytes(before)

            reset()
            result = worker.publish(questions, before, {})
            assert result["added_scores"] == 19 and result["local_valid_scores"] == 38
            assert shared.read_bytes() == complete and cache.read_bytes() == complete
            checks.append("Valid records append once, preserving all previous record bytes and local input")

            null = json.loads(real_records[37])
            null["result"].update({worker.FIELD: None, "Justification": None})
            reset(b"".join(real_records[i] for i in range(37)) + (json.dumps(null) + "\n").encode())
            result = worker.publish(questions, before, {})
            assert result["added_scores"] == 18 and result["local_null_indexes_excluded"] == [37]
            assert shared.read_bytes() == b"".join(real_records[i] for i in range(37))
            checks.append("An official null record is excluded and disclosed for later full scoring")

            bad = json.loads(real_records[0])
            bad["result"]["Justification"] += " changed fixture"
            variants = [complete + real_records[0],
                        (json.dumps(bad) + "\n").encode() + b"".join(real_records[i] for i in range(1, 38))]
            for invalid in variants:
                reset(invalid)
                try:
                    worker.publish(questions, before, {})
                except ValueError:
                    pass
                else:
                    raise AssertionError("Duplicate or conflicting score was accepted")
                assert shared.read_bytes() == before
            checks.append("Duplicate and conflicting scores fail before changing the canonical cache")

            reset()
            with patch.object(worker, "guard", side_effect=ValueError("Inspected failure no longer present")):
                try:
                    worker.publish(questions, before, {})
                except ValueError:
                    pass
                else:
                    raise AssertionError("Publication continued without the inspected failure")
            assert shared.read_bytes() == before
            checks.append("A closed publication window leaves the canonical cache unchanged")

            reset()
            shared.write_bytes(before + real_records[19])
            changed = shared.read_bytes()
            try:
                worker.publish(questions, before, {})
            except ValueError:
                pass
            else:
                raise AssertionError("A concurrently changed canonical cache was overwritten")
            assert shared.read_bytes() == changed
            checks.append("A concurrent canonical change is preserved and publication is rejected")
    result = {"state": "passed", "checked_at": worker.audit.now(), "checks": checks,
              "api_calls": 0, "real_process_signals": 0, "formal_inputs_modified": False,
              "entrypoint_sha256": worker.audit.sha256(Path(worker.__file__)),
              "validation_sha256": worker.audit.sha256(Path(__file__))}
    worker.audit.write_json(worker.audit.RUN / "parallel_smfa_meaningful_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
