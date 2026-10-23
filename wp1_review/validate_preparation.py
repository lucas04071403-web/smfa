"""CPU checks for identity recovery, score joins, mask evidence and GPU admission."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wp1_review import prepare_wp1 as prep
from wp1_review import run_step0_baselines as runner


def expect_error(call, error=ValueError):
    try:
        call()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def main():
    passed = []
    frame = pd.DataFrame([
        {"ID": pid, "image understanding": [{"Image": "character", "Question": "Face?", "Answer": "A face"}],
         "image memory": [], "text memory": [{"Question": "Name?", "Answer": "Unknown"}]}
        for pid in ("person_a", "person_b")
    ])
    questions = []
    for pid in ("person_a", "person_b"):
        questions += [
            {"question type": "image_understanding", "image_id": pid, "image_type": "character",
             "question": "Face?", "ground_truth": "A face", "generated_answer": "A face"},
            {"question type": "text_memory", "question": "Name?", "ground_truth": "Unknown",
             "generated_answer": "I don't know"},
        ]
    assert [pid for _, pid in prep.annotate_questions(questions, frame)] == ["person_a", "person_a", "person_b", "person_b"]
    bad = [dict(q) for q in questions]
    bad[2]["image_id"] = "person_a"
    expect_error(lambda: prep.annotate_questions(bad, frame))
    passed.append("Text identities are inherited from source order; wrong image identities are rejected")

    records = [
        {"Question Type": q["question type"], "Image_ID": q.get("image_id"),
         "Question": q["question"], "Generated Answer": q["generated_answer"],
         "Ground Truth": q["ground_truth"], "Factuality Score": value}
        for q, value in zip(questions, (8, 0, 9, 0))
    ]
    assert prep.align_judge_rows(questions, list(reversed(records)), "Factuality Score") == [8, 0, 9, 0]
    bad = [dict(record) for record in records]
    bad[3]["Factuality Score"] = 1
    expect_error(lambda: prep.align_judge_rows(questions, bad, "Factuality Score"))
    passed.append("Unordered judge tuples align; ambiguous text duplicates require indexed caches")

    text = "[multi] masked parameters: 0/100, ratio: 0.0000\n[text]  masked parameters: 0/100, ratio: 0.0000\n"
    assert runner.parse_zero_masks(text)["text"]["masked"] == 0
    expect_error(lambda: runner.parse_zero_masks(text.replace("0/100", "1/100", 1)))
    expect_error(lambda: runner.parse_zero_masks(text.splitlines()[0]))
    passed.append("Zero-mask evidence requires both official log lines and rejects nonzero masks")

    probe = subprocess.CompletedProcess([], 0, json.dumps({"torch": "fixture", "built_cuda": "fixture",
            "cuda_available": True, "device_count": 2, "factuality_client": True, "meaningful_client": True}), "")
    devices = subprocess.CompletedProcess([], 0, "0, gpu-a, A40, 45066, 46068\n1, gpu-b, A40, 4214, 46068\n", "")
    status = {"preflight": {}}
    with patch.object(runner.subprocess, "run", side_effect=[probe, devices]):
        expect_error(lambda: runner.preflight(status, 32768), runner.GPUUnavailable)
    assert status["preflight"]["scoring_configured"]
    passed.append("Two visible GPUs are insufficient when GPU 1 has only 4214 MiB free")

    with tempfile.TemporaryDirectory(prefix="wp1-validation-", dir=ROOT / "wp1_review") as directory:
        folder = Path(directory)
        status = {"synthetic_mfa": {"state": "not_started", "stages": []}}
        command = [sys.executable, "-c", f"print({text!r})"]
        with patch.object(runner, "RUN", folder), patch.object(runner, "check_saved_checkpoint") as checkpoint:
            runner.run_stage(status, "synthetic_mfa", "sculpt", command, 0, folder / "unused.log", check_zero_mask=True)
            runner.run_stage(status, "synthetic_mfa", "sculpt", command, 0, folder / "unused.log", check_zero_mask=True)
            assert checkpoint.call_count == 2
        assert len(status["synthetic_mfa"]["stages"]) == 1
        assert status["synthetic_mfa"]["stages"][0]["duration_seconds"] >= 0
        assert json.loads((folder / "MFA_unsculpted_zero_mask.json").read_text())["state"] == "passed"
        assert not (folder / "sculpt_stats.json").exists()
    passed.append("A sculpt subprocess succeeds with official log evidence only; completed stages are reused")

    partition = prep.load_partition()
    selection = json.loads((ROOT / "wp1_review/memory_construct_answer_selection.json").read_text())
    assert selection["questions"] == len(selection["reference"]) == 30
    assert {q["person_id"] for q in selection["reference"]} <= set(partition["forget"]["F1_development"])
    assert selection["split_manifest_sha256"] == prep.sha256(ROOT / "wp1_review/fixed_person_split.json")
    passed.append("The actual 30-question selection uses only frozen F1 development identities")
    result = {"state": "passed", "checks": passed, "api_calls": 0, "gpu_allocations": 0}
    prep.atomic_json(ROOT / "wp1_review/preparation_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
