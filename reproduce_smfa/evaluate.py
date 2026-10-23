"""Checkpoint each profile while retaining the released generation and metrics."""

import copy
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer

REPO = Path(__file__).resolve().parents[1] / "S-MLLMUn"
sys.path.insert(0, str(REPO))
import eval as official

evaluate_profile = official.evaluate_generation
COLUMNS = {"image_understanding": "image understanding", "image_memory": "image memory", "text_memory": "text memory"}
LABELS = {"image_understanding": "image_understanding", "image_memory": "Image_memory", "text_memory": "Text_memory"}


def expected_questions(rows):
    return [(kind, q["Question"], q["Answer"]) for _, row in rows.iterrows()
            for kind, column in COLUMNS.items() for q in row[column]]


def verify_questions(questions, rows):
    actual = [(q["question type"], q["question"], q["ground_truth"]) for q in questions]
    if actual != expected_questions(rows):
        raise RuntimeError("Generated question coverage differs from the evaluation dataset")


def resumable_evaluation(parquet_file, processor, tokenizer, model, args, mode="default"):
    rows = pd.read_parquet(parquet_file)
    output = Path(args.output_folder)
    output.mkdir(parents=True, exist_ok=True)
    parts = output / ".profiles" / mode
    parts.mkdir(parents=True, exist_ok=True)
    questions = []
    for index in range(len(rows)):
        profile = rows.iloc[index:index + 1]
        folder = parts / f"{index:04d}"
        result_file = folder / f"{mode}_generation_results.json"
        if not result_file.exists():
            folder.mkdir(parents=True, exist_ok=True)
            temporary_input = folder / "input.parquet"
            profile.to_parquet(temporary_input, index=False)
            local_args = copy.copy(args)
            local_args.output_folder = str(folder)
            evaluate_profile(str(temporary_input), processor, tokenizer, model, local_args, mode)
            temporary_input.unlink()
        result = json.loads(result_file.read_text())["Generation_Questions"]
        verify_questions(result, profile)
        questions.extend(result)
        progress = {"split": mode, "completed_profiles": index + 1, "total_profiles": len(rows), "completed_questions": len(questions)}
        (output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        print(f"CHECKPOINT {mode}: {index + 1}/{len(rows)} profiles; {len(questions)} questions", flush=True)
    verify_questions(questions, rows)
    final = output / f"{mode}_generation_results.json"
    temporary = final.with_suffix(".tmp")
    temporary.write_text(json.dumps({"Generation_Questions": questions}, indent=4) + "\n")
    temporary.replace(final)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    totals = Counter()
    counts = Counter()
    for q in questions:
        kind = q["question type"]
        totals[kind] += scorer.score(q["ground_truth"], q["generated_answer"])["rougeL"].fmeasure
        counts[kind] += 1
    return {f"Average ROUGE-L ({LABELS[kind]})": totals[kind] / counts[kind] for kind in COLUMNS}


if __name__ == "__main__":
    official.evaluate_generation = resumable_evaluation
    official.main()
