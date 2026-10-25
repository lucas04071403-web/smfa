"""Immutable supplement inputs and development-only result locations."""
import json
from pathlib import Path
from datetime import datetime
from wp1_a_stage.common import ROOT, now, atomic_json, sha256, frozen as parent_frozen

OUT = ROOT / 'wp1_supplement'
PARENT = ROOT / 'wp1_a_stage'

def frozen():
    partition = parent_frozen()
    for line in (OUT / 'FREEZE_supplement.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert sha256(OUT / name) == digest, f'Supplement frozen input changed: {name}'
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert sha256(PARENT / 'FREEZE.sha256') == protocol['parent_freeze_sha256']
    assert sha256(PARENT / 'RESULTS.sha256') == protocol['parent_results_sha256']
    return protocol, partition

def after_freeze(timestamp):
    protocol, _ = frozen()
    assert datetime.fromisoformat(timestamp) > datetime.fromisoformat(protocol['frozen_at'])

def result_dir(name):
    if name == 'SMFA' or name in ('Original', 'Base', 'MFA_unsculpted'):
        return PARENT / 'references' / name
    old = PARENT / 'candidates' / name
    return old if (old / 'metrics.json').exists() else OUT / 'candidates' / name

def queue(config, priority=100):
    frozen()
    path = OUT / 'queue' / f'{priority:04d}_{config["name"]}.json'
    if path.exists():
        assert json.loads(path.read_text()) == config
    else:
        atomic_json(path, config)
    return path

def read_metrics(name):
    return json.loads((result_dir(name) / 'metrics.json').read_text())

def summary(name):
    m = read_metrics(name)
    get = lambda label, key='rougeL': m[label]['mean'][key]
    return dict(name=name, forget_combined=get('forget/combined_memory'),
                forget_image=get('forget/image_memory'), forget_text=get('forget/text_memory'),
                retain_image=get('retain/image_memory'), retain_text=get('retain/text_memory'),
                retain_face=get('retain/image_understanding/character'),
                retain_eye=get('retain/image_understanding/ophthalmic'),
                retain_eye_refusal=get('retain/image_understanding/ophthalmic', 'refusal_loose'),
                refusal_loose=get('retain/all', 'refusal_loose'),
                answer_words=get('retain/all', 'answer_words'))
