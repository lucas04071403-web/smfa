"""One persistent model per worker; development-only, per-person resumable outputs."""
import os,sys,json,time,argparse,hashlib,fcntl
from pathlib import Path
from io import BytesIO
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from transformers import AutoProcessor,LlavaOnevisionForConditionalGeneration
from wp1_review.memory_construct import ORIGINAL,BASE_MODEL,official_answer,load_lora_matrices
from wp1_a_stage.common import OUT,dev_frames,questions_for_row,metrics,now,atomic_json,sha256,frozen
from wp1_a_stage.candidates import INITIAL,SPECTRAL,apply

def run(names):
 frozen();guard=json.loads((OUT/'c1_exact_guard.json').read_text());assert guard['state']=='passed'
 assert guard['code_sha256']==sha256(OUT/'candidates.py')
 torch.set_num_threads(4);torch.manual_seed(20261023);torch.backends.cuda.matmul.allow_tf32=False
 frames=dev_frames();configs={x['name']:x for x in INITIAL+SPECTRAL}
 for name in names:
  extra=OUT/'extra_candidates'/f'{name}.json'
  if name not in configs and extra.exists():configs[name]=json.loads(extra.read_text())
 matrices,scales=load_lora_matrices();start=time.monotonic()
 model=LlavaOnevisionForConditionalGeneration.from_pretrained(str(ORIGINAL),torch_dtype=torch.float16,device_map={'':'cuda:0'},low_cpu_mem_usage=True,local_files_only=True).eval()
 processor=AutoProcessor.from_pretrained(str(BASE_MODEL),local_files_only=True);load_seconds=time.monotonic()-start
 code={p.name:sha256(p) for p in [OUT/'evaluate.py',OUT/'candidates.py',OUT/'common.py',ROOT/'wp1_review/memory_construct.py']}
 for name in names:
  cfg=configs[name];out=OUT/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
  lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  identity={'config':cfg,'code_sha256':code,'freeze_sha256':sha256(OUT/'FREEZE.sha256'),'eigen_sha256':sha256(OUT/'statistics/S0_summary.json') if cfg['family'] in ['C2','C3'] else None}
  manifest=out/'manifest.json'
  if manifest.exists():assert json.loads(manifest.read_text())==identity,'Cache identity changed'
  else:atomic_json(manifest,identity)
  if (out/'metrics.json').exists():lock.close();continue
  status=dict(state='constructing',candidate=name,started_at=now(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),load_seconds=load_seconds,completed_questions=0,total_questions=2375)
  atomic_json(out/'status.json',status);build=apply(model,matrices,scales,cfg);atomic_json(out/'construction.json',build)
  t=time.monotonic();qs_all=[];n_new=0
  try:
   for split,frame in frames.items():
    output=[]
    for source_index,row in frame.iterrows():
     expected=questions_for_row(row,split,source_index);cache=out/'people'/split/f'{row.ID}.json'
     if cache.exists():
      qs=json.loads(cache.read_text())['questions'];assert len(qs)==19
      for q,ref in zip(qs,expected):assert all(q[k]==v for k,v in ref.items())
     else:
      face=Image.open(BytesIO(row.character_image['bytes']));eye=Image.open(BytesIO(row.ophthalmic_image['bytes']));qs=[]
      with torch.inference_mode():
       for ref in expected:
        image=None if ref['question type']=='text_memory' else face if ref['image_type']=='character' else eye
        tick=time.monotonic();answer=official_answer(processor,model,ref['question'],image)
        qs.append(dict(ref,generated_answer=answer,answer_tokens=len(processor.tokenizer.encode(answer,add_special_tokens=False)),generation_seconds=time.monotonic()-tick));n_new+=1
      atomic_json(cache,dict(questions=qs))
     output+=qs
     status.update(state='generating',updated_at=now(),completed_questions=len(qs_all)+len(output),new_questions_this_run=n_new,elapsed_generation_seconds=time.monotonic()-t)
     atomic_json(out/'status.json',status)
     if status['completed_questions']%95==0:print(name,status['completed_questions'],'/2375',round(time.monotonic()-t,1),'s',flush=True)
    atomic_json(out/f'{split}_generation_results.json',{'Generation_Questions':output});qs_all+=output
   assert len(qs_all)==2375
   atomic_json(out/'metrics.json',metrics(qs_all));status.update(state='complete',finished_at=now(),generation_seconds=time.monotonic()-t,construction_seconds=build['seconds']);atomic_json(out/'status.json',status)
  except Exception as e:
   status.update(state='failed',error=repr(e),updated_at=now());atomic_json(out/'status.json',status);raise
  finally:lock.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('names',nargs='+');run(p.parse_args().names)
