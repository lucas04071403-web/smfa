"""Adopt live diagnostic collection and share idle GPU 1 with one eval worker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from wp1_supplement.common import OUT, ROOT, atomic_json, frozen, now, sha256


def alive(pid):
    p = Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split()[2] != 'Z'


def main():
    frozen()
    diagnostic_pid = json.loads((OUT / 'diagnostics_launcher.json').read_text())['pid']
    assert alive(diagnostic_pid), 'Expected the existing diagnostic process'
    existing = OUT / 'worker_gpu1.json'
    assert not existing.exists() or not alive(json.loads(existing.read_text())['pid']), 'GPU 1 worker already running'
    with (OUT / 'logs/gpu1.log').open('ab', buffering=0) as log:
        worker = subprocess.Popen(
            [sys.executable, '-u', '-m', 'wp1_supplement.evaluate'], cwd=str(ROOT),
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='1', TOKENIZERS_PARALLELISM='false'),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(OUT / 'worker_gpu1.json', dict(pid=worker.pid, started_at=now(), gpu=1,
                module='wp1_supplement.evaluate', log='logs/gpu1.log'))
    atomic_json(OUT / 'gpu1_concurrency.json', dict(started_at=now(), diagnostic_pid=diagnostic_pid,
                worker_pid=worker.pid, source_sha256=sha256(Path(__file__)),
                reason='Diagnostic accumulation spends most time on CPU and disk; observed GPU usage about 17 GiB, eval about 19 GiB, within 46 GiB.',
                inference_and_candidate_rules_unchanged=True,
                timing_note='GPU 1 candidate and diagnostic wall times overlap; compare per-question and wall times with this scheduling context.'))
    while alive(diagnostic_pid):
        assert worker.poll() is None, 'Concurrent eval worker exited; inspect logs/gpu1.log'
        time.sleep(10)
    diagnostic = json.loads((OUT / 'diagnostics/status.json').read_text())
    assert diagnostic['state'] == 'complete', 'Diagnostics exited before completion'
    record = json.loads((OUT / 'gpu1_concurrency.json').read_text())
    record['diagnostics_finished_at'] = diagnostic['finished_at']
    record['overlap_ended_at'] = now()
    atomic_json(OUT / 'gpu1_concurrency.json', record)
    atomic_json(OUT / 'gpu1_pipeline_status.json', dict(state='candidate_worker', at=now(), pid=worker.pid))
    assert worker.wait() == 0, 'GPU 1 eval worker failed'
    atomic_json(OUT / 'gpu1_pipeline_status.json', dict(state='complete', at=now()))


if __name__ == '__main__':
    main()
