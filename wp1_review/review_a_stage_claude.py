"""CPU-only audit of A-stage review; leaves frozen experiment artifacts intact."""
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rouge_score.rouge_scorer import RougeScorer

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'wp1_a_stage'
DEST = ROOT / 'wp1_review' / 'a_stage_claude_audit.json'
SEED = 20261024
REPLICATES = 20000


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interpolation(xs, ys, target):
    # Same piecewise-linear, sort-by-observed-x convention as frozen A stage.
    result = np.full(len(target), np.nan)
    for i in range(len(target)):
        unique = {}
        for x, y in zip(xs[i], ys[i]):
            unique[x] = max(unique.get(x, -np.inf), y)
        points = sorted(unique.items())
        if points[0][0] <= target[i] <= points[-1][0]:
            result[i] = np.interp(target[i], *np.asarray(points).T)
    return result


def summarize(values):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return dict(valid_replicates=0)
    ci = np.quantile(finite, [.025, .975])
    return dict(valid_replicates=len(finite), unbracketed_replicates=len(values)-len(finite),
                conditional_ci95=ci.tolist(), conditional_halfwidth95=float((ci[1]-ci[0])/2),
                conditional_fraction_above_0_01=float((finite > .01).mean()))


def main():
    selection = read(STAGE / 'selection.json')
    names = list(selection['development_summary'])
    c1 = [name for name in names if name.startswith('C1_')]
    c2 = ['C2_l10', 'C2_l100', 'C2_l1000']
    fdata, rdata, sources, ids_by_split = {}, {}, {}, {}
    scorer = RougeScorer(['rougeL'], use_stemmer=True)
    for name in names:
        directory = STAGE / ('references' if name in ['Original', 'Base', 'SMFA', 'MFA_unsculpted'] else 'candidates') / name
        for split, store in [('forget', fdata), ('retain', rdata)]:
            path = directory / f'{split}_generation_results.json'
            rows = read(path)['Generation_Questions']
            groups = defaultdict(list)
            for q in rows:
                kind = q['question type']
                if (kind.endswith('memory') if split == 'forget' else kind == 'image_memory'):
                    groups[q['person_id']].append(scorer.score(q['ground_truth'], q['generated_answer'])['rougeL'].fmeasure)
            ids = sorted(groups)
            if split in ids_by_split:
                assert ids == ids_by_split[split]
            else:
                ids_by_split[split] = ids
            assert all(len(groups[pid]) == (13 if split == 'forget' else 7) for pid in ids)
            store[name] = np.array([np.mean(groups[pid]) for pid in ids])
            metric = 'forget_combined' if split == 'forget' else 'retain_image'
            assert abs(store[name].mean()-selection['development_summary'][name][metric]) < 1e-10
            sources[str(path.relative_to(ROOT))] = sha(path)
    split = read(ROOT / 'wp1_review/fixed_person_split.json')
    assert ids_by_split['forget'] == sorted(split['forget']['F1_development'])
    assert ids_by_split['retain'] == sorted(split['retain']['development'])
    rng = np.random.default_rng(SEED)
    fd = rng.integers(0, 25, (REPLICATES, 25))
    rd = rng.integers(0, 100, (REPLICATES, 100))
    xs = {name: values[fd].mean(1) for name, values in fdata.items()}
    ys = {name: values[rd].mean(1) for name, values in rdata.items()}
    xcurve = np.column_stack([xs[n] for n in c1])
    ycurve = np.column_stack([ys[n] for n in c1])
    comparisons = {}
    uplifts = []
    for name in ['SMFA', 'C0_k1', 'C0_k10'] + c2:
        uplift = ys[name] - interpolation(xcurve, ycurve, xs[name])
        summary = summarize(uplift)
        original = selection['development_summary']
        baseline = np.interp(original[name]['forget_combined'],
                             [p[0] for p in selection['c1_curve']],
                             [p[1] for p in selection['c1_curve']])
        summary['estimate'] = original[name]['retain_image']-baseline
        comparisons[name] = summary
        if name in c2:
            uplifts.append(uplift)
    matrix = np.column_stack(uplifts)
    valid = np.isfinite(matrix).all(1)
    winners = np.argmax(matrix[valid], axis=1)
    ranking = dict(fully_bracketed_replicates=int(valid.sum()),
                   conditional_selection_frequencies={name: float((winners == i).mean()) for i, name in enumerate(c2)})
    # This measures sampling variation conditional on observed sparse knots, not curvature bias.
    output = dict(at=datetime.now(timezone.utc).isoformat(), seed=SEED, replicates=REPLICATES,
                  scope='Frozen F1 and retain-development only; no GPU or API calls',
                  bootstrap='Independent person resampling between forget/retain; identical draws across models within cohort; reinterpolate each replicate; no extrapolation',
                  limitation='Conditional on existing knots and candidate grid; excludes unknown curvature, selection-process uncertainty and judging noise; bootstrap winner frequencies are not probabilities a lambda is best in the population.',
                  comparisons_vs_c1=comparisons, lambda_ranking=ranking,
                  generation_source_sha256=sources,
                  selection_sha256=sha(STAGE / 'selection.json'))
    import torch
    torch.set_num_threads(2)
    spectra = {}
    for path in sorted((STAGE / 'statistics/eigen').glob('*.pt')):
        spec = torch.load(path, map_location='cpu', weights_only=True)
        values = spec['all_eigenvalues'].numpy()
        r = len(spec['values'])
        spectra[path.stem] = dict(r=r, counts={str(th): dict(full=int((values/values[0] > th).sum()),
                                                           protected=int((values[:r]/values[0] > th).sum())) for th in [.1, .01, .001]})
        del spec
    output['spectral_threshold_counts'] = spectra
    for name in ['FREEZE.sha256', 'RESULTS.sha256']:
        output[name + '_sha256'] = sha(STAGE / name)
    DEST.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: output[key] for key in ['comparisons_vs_c1', 'lambda_ranking']}, indent=2))
    print('Saved aggregate-only audit:', DEST)


if __name__ == '__main__':
    main()
