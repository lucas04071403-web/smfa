"""Publish the completed WP1 supplement from an explicit artifact allowlist."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 1024 * 1024
SUMMARY_PATHS = [
    'protocol.json', 'templates.json', 'rule_numeric_interpretation.json',
    'original_curve_recalculation.json', 'c1_eight_point_selection.json',
    'c1_selection.json', 'c1_bootstrap.json', 'B1_selection_inputs.json',
    'B1_baselines.json', 'B1_operator_selection.json', 'completion_audit.json',
    'workflow_status.json', 'workflow_unit_checks.json',
    'operator_comparison_correction.json', 'gpu1_concurrency.json',
    'appearance/summary.json', 'appearance/appearance_holdout/sensitivity.json',
    'diagnostics/coverage.json', 'diagnostics/spectral_threshold_counts.json',
    'diagnostics/scope_extension.json', 'diagnostics/unit_checks.json',
    'diagnostics/pilot/validation.json', 'diagnostics/status.json',
    'stability/selection.json', 'stability/results.json', 'stability/status.json',
    'stability/half_A/summary.json', 'stability/half_B/summary.json',
    'recovery/latest.json', 'recovery/workflow_unit_checks.json',
    'implementation_archive_before_6p2_correction/workflow_unit_checks.json',
]
HASH_PATHS = [
    'FREEZE_supplement.sha256', 'RESULTS_supplement.sha256',
    'B1_selection_inputs.sha256', 'OPERATOR_COMPARISON.sha256',
    'stability/SELECTION.sha256',
]
FIGURES = ['c1_enrichment.png', 'c1_enrichment.svg',
           'supplement_tradeoffs.png', 'supplement_tradeoffs.svg']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def hash_qa(value):
    """Keep sample identity and hashes without dataset question/answer text."""
    if isinstance(value, dict):
        return {k + '_sha256' if k in {'question', 'ground_truth', 'answer'} else k:
                digest(v.encode()) if k in {'question', 'ground_truth', 'answer'} else hash_qa(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [hash_qa(v) for v in value]
    return value


def sanitized_selection(value):
    value = hash_qa(value)
    value['publication_note'] = 'Dataset question and answer text replaced by SHA-256; original frozen input remains on the experiment host.'
    return value


def sanitized_similarity(value):
    for check in value['checks']:
        for nearest in check['nearest']:
            nearest['eval_question_sha256'] = digest(nearest.pop('question').encode())
    value['publication_note'] = 'Nearest dataset-question text omitted; authored templates, similarities and hashes retained.'
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    source = parser.parse_args().source.resolve()
    if source == ROOT:
        raise SystemExit('Use the separate experiment workspace as source')
    stage = source / 'wp1_supplement'
    audit = json.loads((stage / 'completion_audit.json').read_text())
    workflow = json.loads((stage / 'workflow_status.json').read_text())
    if audit['state'] != 'passed' or workflow['state'] != 'complete':
        raise SystemExit('Only a successfully completed supplement may be archived')
    # Check original frozen/result manifests before exporting any bytes.
    for base, name in [(source / 'wp1_a_stage', 'FREEZE.sha256'),
                       (source / 'wp1_a_stage', 'RESULTS.sha256'),
                       (stage, 'FREEZE_supplement.sha256'),
                       (stage, 'RESULTS_supplement.sha256')]:
        for line in (base / name).read_text().splitlines():
            expected, relative = line.split('  ', 1)
            if digest((base / relative).read_bytes()) != expected:
                raise ValueError(f'Original artifact changed: {base / relative}')
    manifest_path = ROOT / 'PUBLISH_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    records = {item['path']: item for item in manifest['source_files']}
    copied = []

    def save(relative, transform=None, note=None, destination=None, text_transform=None):
        src = source / relative
        if not src.is_file() or src.is_symlink():
            raise ValueError(f'Missing or symlink source: {relative}')
        original = src.read_bytes()
        content = original if transform is None else (
            json.dumps(transform(json.loads(original)), ensure_ascii=False, indent=2) + '\n'
        ).encode()
        if text_transform is not None:
            content = text_transform(original.decode()).encode()
        if len(content) > MAX_BYTES:
            raise ValueError(f'Selected file exceeds 1 MiB: {relative}')
        output = destination or relative
        dst = ROOT / output
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(content)
        shutil.copymode(src, dst)
        record = dict(path=output, source_sha256=digest(original),
                      published_sha256=digest(content), bytes=len(content),
                      transformations=[] if note is None else [note])
        if output != relative:
            record['source_path'] = relative
        records[output] = record
        copied.append(output)

    for directory in [stage, stage / 'implementation_archive_before_6p2_correction',
                      source / 'wp1_review']:
        for pattern in ['*.py', '*.md']:
            for src in sorted(directory.glob(pattern)):
                relative = str(src.relative_to(source))
                if relative == 'wp1_review/step0_image_understanding_report.md':
                    save(relative,
                         text_transform=lambda text: text.replace(
                             'Base 和 MFA_unsculpted 的完整结果产生后，重新运行 prepare_wp1.py --analyze-existing 即可补入。',
                             '四个模型均已完成汇总；逐题输出保留在本地，本仓库只公开汇总指标。'),
                         note='Retained the existing publication completion note instead of the stale local future-tense note')
                else:
                    save(relative)
    for name in SUMMARY_PATHS + HASH_PATHS:
        save('wp1_supplement/' + name)
    for name in FIGURES:
        save('wp1_supplement/figures/' + name)
    for name in ['diagnostic_selection.json'] + [f'diagnostics/{d}/summary.json' for d in ['pilot', 'D_fm', 'D_fi', 'D_rh']]:
        save('wp1_supplement/' + name, sanitized_selection,
             'Replaced dataset question and ground_truth strings with SHA-256; retained sample identity and numerical diagnostics')
    save('wp1_supplement/template_similarity.json', sanitized_similarity,
         'Removed nearest evaluation-question text; retained authored templates, similarities and text hashes')
    for child in sorted((stage / 'candidates').iterdir()):
        if child.is_dir():
            for name in ['manifest.json', 'construction.json', 'status.json', 'metrics.json']:
                save(str((child / name).relative_to(source)))
    for dirname in ['matching', 'plans']:
        for src in sorted((stage / dirname).glob('*.json')):
            save(str(src.relative_to(source)))
    for model in ['Original', 'SMFA', 'MFA_unsculpted']:
        for name in ['manifest.json', 'status.json']:
            save(f'wp1_supplement/appearance/appearance_holdout/{model}/{name}')
    # The complete numeric response table is >1 MiB. Split by dataset without dropping rows.
    response_path = 'wp1_supplement/diagnostics/response_ratios.json'
    response = json.loads((source / response_path).read_text())
    datasets = sorted({row['dataset'] for row in response['rows']})
    for dataset in datasets:
        def select_dataset(value, selected=dataset):
            value['rows'] = [row for row in value['rows'] if row['dataset'] == selected]
            value['publication_note'] = 'Exact numeric rows for this dataset; the union of dataset files reconstructs the original rows.'
            return value
        save(response_path, select_dataset, 'Partitioned complete numerical response rows by dataset to keep each file below 1 MiB',
             f'wp1_supplement/diagnostics/response_ratios_{dataset}.json')
    recovery = json.loads((stage / 'recovery/latest.json').read_text())
    run = Path(recovery['directory'])
    if run.parent != stage / 'recovery':
        raise ValueError('Recovery directory must belong to this supplement')
    for name in ['cache_audit.json', 'completed_metrics_recheck.json', 'runtime.json']:
        save(str((run / name).relative_to(source)))
    for pattern in ['before/workflow_status.json', 'before/candidates/*/status.json',
                    'before/candidates/*/construction.json']:
        for src in sorted(run.glob(pattern)):
            save(str(src.relative_to(source)))

    manifest['updated_at'] = datetime.now(timezone.utc).isoformat()
    manifest['source_files'] = sorted(records.values(), key=lambda item: item['path'])
    manifest['note'] = ('Historical frozen/result hashes describe original local artifacts. Published selection and diagnostic '
                        'copies replace dataset QA with hashes; all transforms and response-table partitions are recorded per file. '
                        'Models, datasets, raw generations, judge responses and activation matrices stay local. '
                        'The archive records the original completed B1 selection; proposed post-B1 protocol changes are not applied.')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    # Keep the allowlist explicit, generated from exactly the copied evidence paths.
    ignore = ROOT / '.gitignore'
    marker = '# Completed WP1 supplement: explicit published evidence allowlist.\n'
    prefix = ignore.read_text().split(marker)[0].rstrip() + '\n\n'
    exceptions = [f'!/{name}' for name in sorted(copied)
                  if Path(name).suffix in {'.json', '.sha256', '.png', '.svg'}]
    ignore.write_text(prefix + marker + '\n'.join(exceptions) + '\n')
    print(json.dumps(dict(copied=len(copied), source_files=len(records),
                          completed_candidates=audit['n_new_candidates'],
                          selected_operator=audit['selected_operator']), indent=2))


if __name__ == '__main__':
    main()
