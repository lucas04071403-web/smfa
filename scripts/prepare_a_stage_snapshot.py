"""Copy an explicit source/summary allowlist; never copy model or dataset trees."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 1024 * 1024
SUMMARY_PATHS = [
    'protocol.json', 'b2r_ids.json', 'templates.json', 'selection.json',
    'A_closeout.json', 'completion_audit.json', 'analysis_checks.json',
    'c1_exact_guard.json', 'operator_unit_checks.json', 'spectral_validation.json',
    'duplicate_activation_diagnostic.json', 'implementation_events.json', 'status.json',
    'statistics/S0_summary.json', 'pilot/S0_summary.json', 'pilot/cost_estimate.json',
    'factuality/run.json', 'factuality/status.json',
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if source == ROOT:
        raise SystemExit('Source must be the separate experiment workspace')
    manifest_path = ROOT / 'PUBLISH_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    records = {item['path']: item for item in manifest['source_files']}
    copied = []

    def save(relative, transform=None, note=None):
        src = source / relative
        if not src.is_file() or src.is_symlink():
            raise ValueError(f'Missing or symlink source: {relative}')
        original = src.read_bytes()
        content = original if transform is None else (
            json.dumps(transform(json.loads(original)), ensure_ascii=False, indent=2) + '\n'
        ).encode()
        if len(content) > MAX_BYTES:
            raise ValueError(f'Selected file exceeds 1 MiB: {relative}')
        dst = ROOT / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(content)
        shutil.copymode(src, dst)
        records[relative] = dict(path=relative, source_sha256=digest(original),
                                 published_sha256=digest(content), bytes=len(content),
                                 transformations=[] if note is None else [note])
        copied.append(relative)

    # Refresh existing source files; preserve previously sanitized documents/JSON.
    for relative, record in list(records.items()):
        src = source / relative
        if src.is_file() and src.suffix in {'.py', '.sh', '.md', '.yml', '.yaml'} and not record['transformations']:
            save(relative)
    for src in sorted(source.glob('*.md')):
        if src.name not in records:
            save(src.name)
    stage = source / 'wp1_a_stage'
    for pattern in ['*.py', '*.md']:
        for src in sorted(stage.glob(pattern)):
            save(str(src.relative_to(source)))
    def compact_statistics(value):
        for key in ['prefix_checks', 'visual_checks']:
            value[key + '_count'] = len(value.pop(key))
        value['publication_note'] = 'Detailed activation checks remain local; aggregate maxima, counts and all group statistics retained.'
        return value

    for relative in SUMMARY_PATHS:
        if relative.endswith('S0_summary.json'):
            save('wp1_a_stage/' + relative, compact_statistics,
                 'Omitted detailed activation check arrays; retained counts, aggregate maxima and all group statistics')
        else:
            save('wp1_a_stage/' + relative)
    for relative in ['FREEZE.sha256', 'RESULTS.sha256', 'A_tradeoff.png']:
        save('wp1_a_stage/' + relative)

    def selected_without_qa(value):
        for sample in value['selected']:
            for key in ['question', 'answer']:
                sample[key + '_sha256'] = digest(sample.pop(key).encode())
        value['publication_note'] = 'Dataset question/answer text omitted; source indices and hashes retained. Original stays on experiment host.'
        return value

    save('wp1_a_stage/b1_selection.json', selected_without_qa,
         'Removed dataset questions and answers; retained source indices and text hashes')

    def similarity_without_eval_questions(value):
        for check in value['checks']:
            for nearest in check['nearest']:
                nearest['eval_question_sha256'] = digest(nearest.pop('question').encode())
        value['publication_note'] = 'Nearest evaluation-question text omitted; similarity scores and text hashes retained.'
        return value

    save('wp1_a_stage/template_similarity.json', similarity_without_eval_questions,
         'Removed nearest evaluation-question text; retained authored templates, similarity values and hashes')
    # Only selected configurations, timing, masks and aggregate metrics from candidate trees.
    for directory in [stage / 'candidates', stage / 'references']:
        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue
            names = ['manifest.json', 'construction.json', 'status.json', 'metrics.json'] if directory.name == 'candidates' else ['provenance.json', 'metrics.json']
            for name in names:
                save(str((child / name).relative_to(source)))

    manifest['updated_at'] = datetime.now(timezone.utc).isoformat()
    manifest['source_files'] = sorted(records.values(), key=lambda item: item['path'])
    manifest['note'] = ('Historical/frozen hashes describe original experiment artifacts, including omitted local evidence. '
                        'b1_selection.json and template_similarity.json are sanitized publication copies; '
                        'published_sha256 below describes uploaded bytes. No raw dataset QA, generations or judge responses are included.')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(source_files=len(records), copied=len(copied),
                          sanitized=['wp1_a_stage/b1_selection.json', 'wp1_a_stage/template_similarity.json']),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
