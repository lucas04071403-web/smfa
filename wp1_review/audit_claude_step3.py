"""CPU-only review of step-3 statistical proposals using frozen step-0 scores.

This is a planning audit, not a new candidate experiment or an equivalence test.
It never changes partitions, generates answers, or requests judge API scores.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from prepare_wp1 import EVAL, DATA, OUT, annotate_questions, load_judge_scores, rows


MODELS = ("Base", "Original", "MFA_unsculpted", "SMFA")
KINDS = ("image_memory", "text_memory")
SEED = 20261023
BOOTSTRAPS = 20000


def ratio_gap(x):
    """x is persons x models; means are taken before normalizing."""
    means = x.mean(axis=0)
    return (means[3] - means[2]) / (means[1] - means[0])


def bootstrap(x, seed):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(x), size=(BOOTSTRAPS, len(x)))
    means = x[draws].mean(axis=1)
    denominators = means[:, 1] - means[:, 0]
    if np.any(denominators <= 0):
        raise ValueError("Nonpositive denominator in bootstrap; ratio interval is not reliable")
    gaps = (means[:, 3] - means[:, 2]) / denominators
    estimate = float(ratio_gap(x))
    lo, hi = np.quantile(gaps, [.05, .95])
    first_lo, first_hi = np.quantile(gaps[:2000], [.05, .95])
    return {
        "n_people": len(x), "difference_pp": 100 * estimate,
        "ci90_pp": [100 * float(lo), 100 * float(hi)],
        "halfwidth_pp": 50 * float(hi - lo),
        "max_distance_from_estimate_pp": 100 * float(max(estimate-lo, hi-estimate)),
        "first_2000_resamples_halfwidth_pp": 50 * float(first_hi-first_lo),
        "bootstrap_denominator_min": float(denominators.min()),
        "h_le_2_5pp": bool(50 * (hi-lo) <= 2.5),
    }


def main():
    manifest_path = OUT / "fixed_person_split.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    frame = rows(DATA / "eval_data/forget_5/train-00000-of-00001.parquet")
    ids = sorted(frame["ID"].tolist())
    positions = {pid: i for i, pid in enumerate(ids)}
    arrays = np.zeros((50, 4, 2), dtype=np.float64)
    provenance = {}
    for mi, model in enumerate(MODELS):
        folder = EVAL / model
        path = folder / "forget_generation_results.json"
        questions = json.loads(path.read_text())["Generation_Questions"]
        annotated = annotate_questions(questions, frame)
        scores, source = load_judge_scores(folder, "forget", questions, "factuality", "Factuality Score")
        assert source["complete"]
        buckets = {(pid, kind): [] for pid in ids for kind in KINDS}
        for (question, pid), score in zip(annotated, scores):
            if question["question type"] in KINDS:
                buckets[pid, question["question type"]].append(score)
        for pid in ids:
            for ki, kind in enumerate(KINDS):
                assert len(buckets[pid, kind]) == (7 if ki == 0 else 6)
                arrays[positions[pid], mi, ki] = np.mean(buckets[pid, kind])
        provenance[model] = {"judge": source, "generation_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    combined = (7 * arrays[:, :, 0] + 6 * arrays[:, :, 1]) / 13
    values = {"image_memory": arrays[:, :, 0], "text_memory": arrays[:, :, 1], "memory_13_questions": combined}
    denominators = {}
    for kind, array in values.items():
        denom = array[:, 1] - array[:, 0]
        safe = denom != 0
        individual_gaps = (array[safe, 3] - array[safe, 2]) / denom[safe]
        denominators[kind] = {
            "person_denominator_min": float(denom.min()), "person_denominator_max": float(denom.max()),
            "nonpositive_count": int((denom <= 0).sum()),
            "cohort_ratio_gap_pp": 100 * float(ratio_gap(array)),
            "mean_of_person_ratio_gaps_pp": 100 * float(individual_gaps.mean()) if safe.all() else None,
        }

    cohorts = {"F1_development": manifest["forget"]["F1_development"],
               "F2_test": manifest["forget"]["F2_test"], "all_50": ids}
    intervals = {}
    for cohort, members in cohorts.items():
        idx = [positions[pid] for pid in members]
        intervals[cohort] = {kind: bootstrap(array[idx], SEED + ci) for ci, (kind, array) in enumerate(values.items())}

    assert manifest_path.read_bytes() == manifest_bytes
    output = {
        "purpose": "Existing-baseline-only planning audit; not a new candidate equivalence test",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "bootstrap_repeats": BOOTSTRAPS,
        "resampling": "paired by person, jointly resampling Base/Original/MFA/SMFA, recomputing ratio of cohort means",
        "interval": "percentile 90%, h=(upper-lower)/2; no clipping; conditional on saved judge outputs",
        "combined_endpoint": "13-question mean score per person, weights 7/13 image and 6/13 text, then cohort normalization",
        "split_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "individual_denominator_audit": denominators, "intervals": intervals, "provenance": provenance,
    }
    (OUT / "claude_step3_statistical_audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"denominators": denominators, "intervals": intervals}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
