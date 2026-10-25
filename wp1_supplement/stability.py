"""Optional, outcome-independent split-half diagnostic of B1 text moments."""
import gc
import json
import os
from pathlib import Path
import random
import time

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from wp1_a_stage.activation_stats import make_inputs, spectrum
from wp1_a_stage.common import DATA, rows
from wp1_review.memory_construct import BASE_MODEL, ORIGINAL
from wp1_supplement.common import OUT, PARENT, atomic_json, frozen, now, sha256

DEST = OUT / 'stability'


def prepare():
    samples = json.loads((PARENT / 'b1_selection.json').read_text())['selected']
    rng = random.Random(20261023)
    halves = [[], []]
    for kind in sorted({s['type'] for s in samples}):
        indices = [i for i, s in enumerate(samples) if s['type'] == kind]
        rng.shuffle(indices)
        first = min(range(2), key=lambda h: len(halves[h]))
        for j, index in enumerate(indices):
            halves[(first+j) % 2].append(index)
    assert len(halves[0]) == len(halves[1]) == 64
    assert sorted(halves[0] + halves[1]) == list(range(128))
    selection = dict(frozen_at=now(), seed=20261023, halves=halves,
                     stratification='Sample type, shuffled then alternated; 64 prompts per half.',
                     b1_selection_sha256=sha256(PARENT/'b1_selection.json'),
                     code_sha256=sha256(Path(__file__)),
                     calibration_content='Same B1 prompts and reference answers used by original calibration statistics.',
                     diagnostic_only=True, changes_selection_rules=False,
                     caveat='Sample halves may share people. This is a descriptive stability check, not a confidence interval.')
    path = DEST / 'selection.json'
    if path.exists():
        saved = json.loads(path.read_text())
        assert saved['halves'] == halves and saved['code_sha256'] == selection['code_sha256']
    else:
        atomic_json(path, selection)
        (DEST/'SELECTION.sha256').write_text(sha256(path)+'  selection.json\n')
    return samples, halves


class TextCollector:
    def __init__(self, model):
        self.acc = {}; self.handles = []; self.text = None
        for layer in range(28):
            block = model.model.language_model.layers[layer]
            for group, module in [('attn', block.self_attn.q_proj), ('o', block.self_attn.o_proj), ('mlp', block.mlp.gate_proj)]:
                key = f'layer_{layer:02d}_{group}'
                self.acc[key] = dict(T=torch.zeros(3584,3584), n_t=0, prefix=None)
                self.handles.append(module.register_forward_pre_hook(self.hook(key)))

    def hook(self, key):
        def fn(module, args):
            x = args[0][0].detach(); s = self.acc[key]
            if s['prefix'] is None: s['prefix'] = x[:2].float().cpu()
            t = x[self.text].float()
            assert torch.isfinite(t).all()
            s['T'].add_((t.T@t).cpu()); s['n_t'] += len(t)
        return fn

    def close(self):
        for h in self.handles: h.remove()


def collect(model, processor, samples, indices, byid, half):
    dest = DEST / f'half_{half}'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'summary.json').exists(): return
    collector = TextCollector(model); start = time.monotonic(); records = []
    torch.cuda.reset_peak_memory_stats()
    try:
        for i, index in enumerate(indices):
            sample = samples[index]; inp = make_inputs(processor, sample, byid)
            ids = inp.input_ids[0]
            assert ids[:2].tolist() == [151644,872]
            text = ids != model.config.image_token_id; text[:2] = False
            collector.text = text.to('cuda')
            with torch.inference_mode():
                result = model.model(**inp.to('cuda'), use_cache=False, return_dict=True)
                del result
            torch.cuda.synchronize(); inp.to('cpu'); del inp
            records.append(dict(index=index, n_text=int(text.sum()), type=sample['type']))
            atomic_json(DEST/'progress.json', dict(at=now(), phase=f'half_{half}', completed=i+1, total=64,
                        seconds=time.monotonic()-start))
            print('text stability',half,i+1,'/64',flush=True)
    finally:
        collector.close()
    for key, s in collector.acc.items():
        assert torch.isfinite(s['T']).all()
        torch.save(s, dest/f'{key}.pt')
    atomic_json(dest/'summary.json',dict(state='complete', at=now(), samples=64, records=records,
                seconds=time.monotonic()-start, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                text_tokens=next(iter(collector.acc.values()))['n_t']))
    del collector; gc.collect(); torch.cuda.empty_cache()


def analyze():
    result = []
    for layer in range(28):
        for group in ('attn','o','mlp'):
            key = f'layer_{layer:02d}_{group}'; matrices = []; specs = []; counts = []
            for half in ('A','B'):
                s = torch.load(DEST/f'half_{half}'/f'{key}.pt',map_location='cpu',weights_only=True)
                matrix = s['T'] + s['prefix'].T@s['prefix']
                matrices.append(matrix.to('cuda')); counts.append(s['n_t']+2)
                spec,r = spectrum(matrices[-1]); specs.append((spec['vectors'].to('cuda'),r))
            va,ra = specs[0]; vb,rb = specs[1]; ca,cb = matrices
            overlap = float((va.T@vb).square().sum())
            row=dict(group=key, rank_A=ra,rank_B=rb,text_tokens_including_single_prefix=counts,
                     overlap_normalized_by_smaller_rank=overlap/min(ra,rb),
                     overlap_normalized_by_larger_rank=overlap/max(ra,rb),
                     A_basis_coverage_on_B=float(((cb@va)*va).sum()/torch.trace(cb)),
                     B_basis_coverage_on_A=float(((ca@vb)*vb).sum()/torch.trace(ca)))
            assert all(-1e-5 <= row[k] <= 1.00001 for k in row if 'coverage' in k or 'overlap_' in k)
            result.append(row)
            atomic_json(DEST/'analysis_progress.json',dict(at=now(),completed_groups=len(result),total=84,last_group=key))
            del matrices,specs,ca,cb,va,vb;gc.collect();torch.cuda.empty_cache()
    atomic_json(DEST/'results.json',dict(state='complete',at=now(),groups=result,tau=.9,
                definition='Each half uses uncentered (T+single-prefix second moment). Eigenvectors span 90% text energy, as in S1 text component.',
                comparison='Cross-half captured energy and principal-subspace overlap; dimensional differences reported.',
                caveat='Descriptive fixed sample split. It does not quantify complete S1 stability or population uncertainty; does not select candidates or thresholds.',
                selection_sha256=sha256(DEST/'selection.json')))


def main():
    frozen();DEST.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20261023);torch.backends.cuda.matmul.allow_tf32=False
    samples,halves = prepare()
    atomic_json(DEST/'status.json',dict(state='running',started_at=now(),pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    frame=rows(DATA/'unlearning_data/retain_few_shot_5/train-00000-of-00001.parquet')
    byid={row.ID:row for _,row in frame.iterrows()}
    model=LlavaOnevisionForConditionalGeneration.from_pretrained(str(ORIGINAL),torch_dtype=torch.float16,
            device_map={'':'cuda:0'},low_cpu_mem_usage=True,local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(str(BASE_MODEL),local_files_only=True)
    for half,indices in zip(('A','B'),halves):collect(model,processor,samples,indices,byid,half)
    del model,processor,byid;gc.collect();torch.cuda.empty_cache()
    analyze()
    atomic_json(DEST/'status.json',dict(state='complete',finished_at=now(),code_sha256=sha256(Path(__file__))))


if __name__=='__main__':main()
