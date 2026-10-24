"""Verify completed A artifacts against the frozen inputs and score tuples."""
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wp1_a_stage.common import OUT, atomic_json, dev_frames, frozen, now, sha256
from wp1_a_stage.candidates import INITIAL, SPECTRAL
from wp1_review.prepare_wp1 import judge_key


def main():
    partition = frozen()
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert sha256(ROOT / 'SMFA_WP1_experiment_design_v3.md') == protocol['design_sha256']
    expected = {}
    for split, frame in dev_frames().items():
        expected[split] = {
            (row.ID, kind, q.get('Image'), q['Question'], q['Answer'])
            for _, row in frame.iterrows()
            for kind, column in [('image_understanding', 'image understanding'),
                                 ('image_memory', 'image memory'), ('text_memory', 'text memory')]
            for q in row[column]
        }
    scope = {}
    generation_hashes = {}
    for config in INITIAL + SPECTRAL:
        name = config['name']
        directory = OUT / 'candidates' / name
        status = json.loads((directory / 'status.json').read_text())
        manifest = json.loads((directory / 'manifest.json').read_text())
        assert status['state'] == 'complete' and status['completed_questions'] == 2375
        assert datetime.fromisoformat(status['started_at']) > datetime.fromisoformat(protocol['frozen_at'])
        assert manifest['config'] == config and manifest['freeze_sha256'] == sha256(OUT / 'FREEZE.sha256')
        for filename, digest in manifest['code_sha256'].items():
            path = ROOT / 'wp1_review' / filename if filename == 'memory_construct.py' else OUT / filename
            assert sha256(path) == digest, (name, filename)
        construction = json.loads((directory / 'construction.json').read_text())
        assert construction['layers'] == 274
        counts = {}
        generation_hashes[name] = {}
        for split, desired in expected.items():
            path = directory / f'{split}_generation_results.json'
            questions = json.loads(path.read_text())['Generation_Questions']
            observed = {(q['person_id'], q['question type'], q.get('image_type'),
                         q['question'], q['ground_truth']) for q in questions}
            assert observed == desired and len(questions) == len(desired)
            people = Counter(q['person_id'] for q in questions)
            assert set(people.values()) == {19}
            for q in questions:
                assert isinstance(q['generated_answer'], str)
                assert q['image_id'] == (None if q['question type'] == 'text_memory' else q['person_id'])
                assert q['split'] == split
            counts[split] = {'questions': len(questions), 'people': len(people),
                             'types': dict(Counter(q['question type'] for q in questions))}
            generation_hashes[name][split] = sha256(path)
        scope[name] = counts
    selection = json.loads((OUT / 'selection.json').read_text())
    assert selection['state'] == 'selected'
    for point in selection['comparisons']:
        curve = selection['c1_curve']
        x = point['forget_combined_rouge']
        left, right = next((a, b) for a, b in zip(curve, curve[1:]) if a[0] <= x <= b[0])
        reference = left[1] + (right[1] - left[1]) * (x - left[0]) / (right[0] - left[0])
        assert abs(reference - point['c1_interpolated_rouge']) < 1e-12
        assert abs(point['retain_image_rouge'] - reference - point['uplift']) < 1e-12
    winner = max(selection['comparisons'], key=lambda p: (p['uplift'], -p['lambda_rel']))
    assert selection['selected_lambda'] == winner['lambda_rel'] == 10
    scoring = json.loads((OUT / 'factuality/scores.json').read_text())
    close = json.loads((OUT / 'A_closeout.json').read_text())
    assert scoring['state'] == close['state'] == 'complete'
    assert len(scoring['records']) == 975
    assert close['scores_sha256'] == sha256(OUT / 'factuality/scores.json')
    assert close['selection_sha256'] == sha256(OUT / 'selection.json')
    assert scoring['run']['official_scorer_sha256'] == sha256(ROOT / 'S-MLLMUn/eval_fact.py')
    assert scoring['run']['runner_sha256'] == sha256(OUT / 'judge.py')
    by_model = defaultdict(dict)
    api_models = Counter()
    timestamps = []
    for record in scoring['records']:
        q = record['question']
        key = (q['person_id'], q['question type'], q['question'])
        assert q['person_id'] in partition['forget']['F1_development']
        assert q['question type'] in ['image_memory', 'text_memory']
        assert key not in by_model[record['model']]
        digest = hashlib.sha256(json.dumps(q, sort_keys=True).encode()).hexdigest()
        assert digest == record['digest']
        assert judge_key(record['result'], scored=True) == judge_key(q)
        score = record['result']['Factuality Score']
        assert type(score) is int and 0 <= score <= 10
        by_model[record['model']][key] = score
        assert record['api_responses']
        for response in record['api_responses']:
            api_models[response['model']] += 1
            timestamps.append(response['at'])
    assert set(by_model) == set(selection['factuality_candidates'])
    assert all(set(items) == set(by_model['SMFA']) and len(items) == 325 for items in by_model.values())
    factuality_means = {
        model: {kind: float(np.mean([score for (_, question_type, _), score in items.items()
                                    if kind == 'combined_memory' or question_type == kind]))
                for kind in ['image_memory', 'text_memory', 'combined_memory']}
        for model, items in by_model.items()
    }
    for model in selection['factuality_candidates'][1:]:
        for kind, stats in close['comparisons'][model].items():
            estimate = 100 * (factuality_means[model][kind] - factuality_means['SMFA'][kind]) / stats['denominator_mean']
            assert abs(estimate - stats['estimate_pp']) < 1e-10
            assert abs((stats['ci90_pp'][1] - stats['ci90_pp'][0]) / 2 - stats['halfwidth_pp']) < 1e-12
    h = max(p['combined_memory']['halfwidth_pp'] for p in close['comparisons'].values())
    assert close['planning_h_pp'] == h
    assert close['margin_pp'] == min(5, math.ceil(2 * max(0, h - 2.5)) / 2)
    summary = json.loads((OUT / 'statistics/S0_summary.json').read_text())
    assert summary['state'] == 'complete' and summary['samples'] == 128 and len(summary['groups']) == 84
    assert sha256(OUT / 'activation_stats.py') == summary['code_sha256']
    assert sha256(OUT / 'b1_selection.json') == summary['source_selection_sha256']
    assert json.loads((OUT / 'spectral_validation.json').read_text())['state'] == 'passed'
    for name in ['operator_unit_checks', 'analysis_checks', 'c1_exact_guard']:
        assert json.loads((OUT / f'{name}.json').read_text())['state'] == 'passed'
    output = dict(state='passed', at=now(), freeze_sha256=sha256(OUT / 'FREEZE.sha256'),
                  new_candidates=10, generated_questions=23750, candidate_scope=scope,
                  no_F2_or_test_retain_in_new_generations=True, generation_sha256=generation_hashes,
                  scored_models={m: len(v) for m, v in by_model.items()}, factuality_means=factuality_means,
                  scoring_api_models=dict(api_models), scoring_window_UTC=[min(timestamps), max(timestamps)],
                  selected_lambda=winner['lambda_rel'], margin_pp=close['margin_pp'],
                  code_sha256=sha256(Path(__file__)))
    atomic_json(OUT / 'completion_audit.json', output)
    print(json.dumps({k: output[k] for k in ['state', 'new_candidates', 'generated_questions', 'scored_models',
                                           'factuality_means', 'selected_lambda', 'margin_pp']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
