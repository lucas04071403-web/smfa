"""Audit saved development caches and resume the existing two-GPU workflow.

Default: CPU audit only. --launch additionally requires two visible CUDA GPUs.
Existing frozen inputs, worker code, candidate caches and results are not edited.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd

from wp1_a_stage.common import questions_for_row
from wp1_review.prepare_wp1 import DATA
from wp1_supplement.common import OUT, PARENT, ROOT, atomic_json, frozen, now, sha256


def read(path):
    return json.loads(path.read_text())


def inventory():
    protocol, partition = frozen()
    for line in (PARENT / 'RESULTS.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert sha256(PARENT / name) == digest, ('Parent result changed', name)
    for name in ('B1_selection_inputs', 'OPERATOR_COMPARISON'):
        for line in (OUT / (name + '.sha256')).read_text().splitlines():
            digest, relative = line.split('  ', 1)
            assert sha256(OUT / relative) == digest, ('Selection input changed', relative)
    b1 = read(OUT / 'B1_selection_inputs.json')
    assert b1['c1_selection_sha256'] == sha256(OUT / 'c1_selection.json')
    expected = {}
    columns = ['ID', 'image understanding', 'image memory', 'text memory']
    for split, key in [('forget', 'F1_development'), ('retain', 'development')]:
        frame = pd.read_parquet(DATA / f'eval_data/{split}_5/train-00000-of-00001.parquet', columns=columns)
        frame['ID'] = frame['ID'].astype(str)
        assert not frame['ID'].duplicated().any()
        frame = frame[frame.ID.isin(partition[split][key])]
        expected[split] = {row.ID: questions_for_row(row, split, index) for index, row in frame.iterrows()}
        assert set(expected[split]) == set(partition[split][key])
    jobs = {read(p)['name']: read(p) for p in sorted((OUT / 'queue').glob('*.json'))}
    candidates = {}
    for path in sorted((OUT / 'candidates').glob('*/manifest.json')):
        directory = path.parent
        manifest = read(path)
        assert manifest['config'] == jobs[directory.name]
        assert manifest['freeze_sha256'] == sha256(OUT / 'FREEZE_supplement.sha256')
        assert manifest['parent_freeze_sha256'] == sha256(PARENT / 'FREEZE.sha256')
        for filename, digest in manifest['code_sha256'].items():
            assert sha256(ROOT / filename) == digest, ('Worker source changed', filename)
        if manifest['config']['family'] in ('C2', 'C3'):
            assert manifest['eigen_sha256'] == sha256(PARENT / 'statistics/S0_summary.json')
        count = 0
        seconds = 0.0
        hashes = {}
        for split in ('forget', 'retain'):
            cached = {}
            for cache in sorted((directory / 'people' / split).glob('*.json')):
                person = cache.stem
                assert person in expected[split], ('Non-development person', directory.name, split)
                questions = read(cache)['questions']
                assert len(questions) == 19
                for q, ref in zip(questions, expected[split][person]):
                    assert all(q[k] == v for k, v in ref.items()), ('Question identity mismatch', directory.name)
                    assert isinstance(q['generated_answer'], str)
                    assert math.isfinite(q['generation_seconds']) and q['generation_seconds'] >= 0
                    assert q['generated_at'] > protocol['frozen_at']
                    if manifest['config']['family'] in ('C2', 'C3'):
                        assert q['generated_at'] > b1['frozen_at']
                        assert q['generated_at'] > read(OUT / 'operator_comparison_correction.json')['frozen_at']
                    seconds += q['generation_seconds']
                cached[person] = questions
                count += len(questions)
                hashes[str(cache.relative_to(OUT))] = sha256(cache)
            export = directory / f'{split}_generation_results.json'
            if export.exists():
                assert set(cached) == set(expected[split])
                ordered = [q for person in expected[split] for q in cached[person]]
                assert read(export)['Generation_Questions'] == ordered
        status = read(directory / 'status.json')
        complete = (directory / 'metrics.json').exists()
        if complete:
            assert count == 2375 and status['state'] == 'complete'
        else:
            assert count <= 2375 and status['completed_questions'] == count
        candidates[directory.name] = dict(
            complete=complete, cached_questions=count, remaining_questions=2375-count,
            last_recorded_state=status['state'], last_recorded_at=status.get('updated_at'),
            cached_generation_seconds=seconds, cache_sha256=hashes,
            manifest_sha256=sha256(path))
    return dict(
        state='passed', checked_at=now(), parent_results_intact=True,
        frozen_inputs_intact=True, worker_sources_intact=True, exact_question_identity_checked=True,
        candidate_cohorts='F1 25 + development retain 100; no new F2/test evaluation',
        selected_lambda=b1['selected_lambda'], candidates=candidates,
        queued_not_started=[name for name in jobs if name not in candidates],
        cached_questions=sum(c['cached_questions'] for c in candidates.values()),
        remaining_queued_questions=sum(c['remaining_questions'] for c in candidates.values())
            + 2375 * sum(name not in candidates for name in jobs),
        remaining_note='Excludes any future bounded bracket, target validation, or C3 extension jobs.')


def gpu_probe():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=20)
        cards = [line.split(',') for line in result.stdout.splitlines()] if result.returncode == 0 else []
        usable = {parts[0].strip(): int(parts[2].strip()) for parts in cards}
        return dict(ready=all(usable.get(i, 0) >= 22000 for i in ('0', '1')),
                    returncode=result.returncode, output=(result.stdout + result.stderr).strip())
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        return dict(ready=False, error=str(error))


def live_processes():
    result = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
            modules = {b'wp1_supplement.evaluate', b'wp1_supplement.controller',
                       b'wp1_supplement.parallel_gpu1', b'wp1_supplement.gpu1_pipeline'}
            if b'-m' in args and modules.intersection(args):
                result.append(int(path.parent.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return result


def launch(directory):
    assert not live_processes(), 'Existing workflow processes found; do not duplicate workers'
    assert not (OUT / 'STOP_WORKERS').exists(), 'STOP_WORKERS exists; inspect completion before restarting'
    assert not (OUT / 'controller_failure.json').exists(), 'Inspect controller failure before restarting'
    assert read(OUT / 'diagnostics/status.json')['state'] == 'complete'
    assert (OUT / 'appearance/summary.json').exists()
    # Check actual OS locks as well as process names before starting new workers.
    for path in [OUT / 'controller.lock', *sorted((OUT / 'candidates').glob('*/run.lock'))]:
        with path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    probe = subprocess.run([sys.executable, '-c',
        'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 2'],
        capture_output=True, text=True, timeout=60)
    assert probe.returncode == 0, 'PyTorch cannot access two CUDA GPUs: ' + probe.stderr[-1000:]
    children = []
    try:
        for label, module, gpu in [('gpu0', 'evaluate', '0'), ('gpu1', 'evaluate', '1'),
                                   ('controller', 'controller', None)]:
            env = os.environ.copy()
            if gpu is not None:
                env['CUDA_VISIBLE_DEVICES'] = gpu
            with (directory / f'{label}.log').open('ab', buffering=0) as log:
                child = subprocess.Popen([sys.executable, '-u', '-m', f'wp1_supplement.{module}'],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
        time.sleep(3)
        assert all(child.poll() is None for child in children), 'A new process exited; inspect recovery logs'
    except BaseException:
        for child in children:
            if child.poll() is None:
                child.terminate()
        raise
    return [child.pid for child in children]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true', help='After successful audit and GPU checks, resume workers and controller')
    args = parser.parse_args()
    recovery = OUT / 'recovery'
    recovery.mkdir(exist_ok=True)
    lock = (recovery / 'resume.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    directory = recovery / now().replace(':', '-')
    directory.mkdir()
    for path in [OUT / 'workflow_status.json', *sorted((OUT / 'candidates').glob('*/status.json')),
                 *sorted((OUT / 'candidates').glob('*/construction.json'))]:
        target = directory / 'before' / path.relative_to(OUT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    audit = inventory()
    atomic_json(directory / 'cache_audit.json', audit)
    runtime = dict(checked_at=now(), gpu=gpu_probe(), visible_existing_processes=live_processes(), launched=False)
    atomic_json(directory / 'runtime.json', runtime)
    if args.launch and runtime['gpu']['ready']:
        runtime.update(launched=True, pids=launch(directory), launched_at=now())
        atomic_json(directory / 'runtime.json', runtime)
    result = dict(directory=str(directory), audit='passed', cached_questions=audit['cached_questions'],
                  remaining_queued_questions=audit['remaining_queued_questions'], **runtime)
    atomic_json(recovery / 'latest.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.launch and not runtime['launched']:
        print('No workers started: two CUDA GPUs are not visible or lack free memory.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
