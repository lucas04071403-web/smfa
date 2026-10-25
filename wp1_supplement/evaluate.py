"""Persistent GPU queue worker. Exact old construction; new, isolated caches."""
import os, sys, json, time, argparse, fcntl, traceback
from pathlib import Path
from io import BytesIO
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from wp1_review.memory_construct import ORIGINAL, BASE_MODEL, official_answer, load_lora_matrices
from wp1_a_stage.common import dev_frames, questions_for_row, metrics
from wp1_a_stage.candidates import apply
from wp1_supplement.common import OUT, PARENT, frozen, after_freeze, now, atomic_json, sha256

def run(idle_timeout=36000):
    frozen()
    guard = json.loads((PARENT / 'c1_exact_guard.json').read_text())
    assert guard['state'] == 'passed' and guard['code_sha256'] == sha256(PARENT / 'candidates.py')
    torch.set_num_threads(4)
    torch.manual_seed(20261023)
    torch.backends.cuda.matmul.allow_tf32 = False
    frames = dev_frames()
    matrices, scales = load_lora_matrices()
    tick = time.monotonic()
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        str(ORIGINAL), torch_dtype=torch.float16, device_map={'': 'cuda:0'},
        low_cpu_mem_usage=True, local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(str(BASE_MODEL), local_files_only=True)
    load_seconds = time.monotonic() - tick
    files = [Path(__file__), OUT/'common.py', PARENT/'candidates.py', PARENT/'common.py', ROOT/'wp1_review/memory_construct.py']
    code = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    idle_since = time.monotonic()
    print('Worker ready', os.environ.get('CUDA_VISIBLE_DEVICES'), now(), flush=True)
    while not (OUT / 'STOP_WORKERS').exists():
        claimed = None
        for job in sorted((OUT / 'queue').glob('*.json')):
            cfg = json.loads(job.read_text())
            out = OUT / 'candidates' / cfg['name']
            if (out / 'metrics.json').exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            lock = (out / 'run.lock').open('w')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            if (out / 'metrics.json').exists():
                lock.close()
                continue
            claimed = cfg, out, lock
            break
        if claimed is None:
            if time.monotonic() - idle_since > idle_timeout:
                break
            time.sleep(3)
            continue
        idle_since = time.monotonic()
        cfg, out, lock = claimed
        frozen()
        assert {str(p.relative_to(ROOT)): sha256(p) for p in files} == code, 'Worker source changed'
        identity = dict(config=cfg, code_sha256=code,
                        freeze_sha256=sha256(OUT/'FREEZE_supplement.sha256'),
                        parent_freeze_sha256=sha256(PARENT/'FREEZE.sha256'),
                        eigen_sha256=sha256(PARENT/'statistics/S0_summary.json') if cfg['family'] in ('C2','C3') else None)
        manifest = out / 'manifest.json'
        if manifest.exists():
            assert json.loads(manifest.read_text()) == identity, 'Cache identity changed'
        else:
            atomic_json(manifest, identity)
        status = dict(state='constructing', candidate=cfg['name'], started_at=now(),
                      pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      load_seconds=load_seconds, total_questions=2375, completed_questions=0)
        after_freeze(status['started_at'])
        atomic_json(out/'status.json', status)
        torch.cuda.reset_peak_memory_stats()
        try:
            build = apply(model, matrices, scales, cfg)
            atomic_json(out/'construction.json', build)
            tick = time.monotonic()
            allq, n_new = [], 0
            status.update(state='generating', generation_started_at=now())
            after_freeze(status['generation_started_at'])
            atomic_json(out/'status.json', status)
            for split, frame in frames.items():
                output = []
                for index, row in frame.iterrows():
                    expected = questions_for_row(row, split, index)
                    cache = out/'people'/split/f'{row.ID}.json'
                    if cache.exists():
                        qs = json.loads(cache.read_text())['questions']
                        assert len(qs) == 19
                        for q, ref in zip(qs, expected):
                            assert all(q[k] == v for k, v in ref.items())
                    else:
                        face = Image.open(BytesIO(row.character_image['bytes']))
                        eye = Image.open(BytesIO(row.ophthalmic_image['bytes']))
                        qs = []
                        with torch.inference_mode():
                            for ref in expected:
                                image = None if ref['question type']=='text_memory' else face if ref['image_type']=='character' else eye
                                t = time.monotonic()
                                answer = official_answer(processor, model, ref['question'], image)
                                qs.append(dict(ref, generated_answer=answer,
                                               answer_tokens=len(processor.tokenizer.encode(answer, add_special_tokens=False)),
                                               generated_at=now(), generation_seconds=time.monotonic()-t))
                                n_new += 1
                        face.close(); eye.close()
                        atomic_json(cache, dict(questions=qs))
                    output.extend(qs)
                    status.update(updated_at=now(), completed_questions=len(allq)+len(output),
                                  new_questions_this_run=n_new, elapsed_generation_seconds=time.monotonic()-tick)
                    atomic_json(out/'status.json', status)
                    if status['completed_questions'] % 95 == 0:
                        print(cfg['name'], status['completed_questions'], '/2375', round(time.monotonic()-tick,1), 's', flush=True)
                atomic_json(out/f'{split}_generation_results.json', dict(Generation_Questions=output))
                allq.extend(output)
            assert len(allq) == 2375
            result = metrics(allq)
            assert result['forget/all']['n_people']==25 and result['retain/all']['n_people']==100
            atomic_json(out/'metrics.json', result)
            status.update(state='complete', finished_at=now(), generation_seconds=time.monotonic()-tick,
                          construction_seconds=build['seconds'], peak_allocated_bytes=torch.cuda.max_memory_allocated())
            atomic_json(out/'status.json', status)
            print('COMPLETE', cfg['name'], status['generation_seconds'], flush=True)
        except BaseException as error:
            status.update(state='failed', updated_at=now(), error=repr(error), traceback=traceback.format_exc())
            atomic_json(out/'status.json', status)
            raise
        finally:
            lock.close()
        idle_since = time.monotonic()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idle-timeout', type=int, default=36000)
    run(parser.parse_args().idle_timeout)
