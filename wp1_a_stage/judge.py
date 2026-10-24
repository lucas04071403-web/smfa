"""Concurrent, interleaved official qwen-plus factuality judging of 3x325 answers."""
import sys,json,time,random,threading,types,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wp1_a_stage.common import OUT,now,atomic_json,sha256,frozen
from wp1_review.run_step0_baselines import configure_network_environment,load_scoring_credentials

def main():
 frozen();configure_network_environment();load_scoring_credentials();sys.path.insert(0,str(ROOT/'S-MLLMUn'));import eval_fact
 from wp1_review.score_json_recovery_runner import parse_json_with_apostrophe_escape_repair
 eval_fact.convert_markdown_json_to_json=parse_json_with_apostrophe_escape_repair
 assert eval_fact.client is not None
 client=eval_fact.client.with_options(timeout=120,max_retries=2);local=threading.local()
 def create(**kwargs):
  t=now();response=client.chat.completions.create(**kwargs)
  local.responses.append(dict(at=t,model=response.model,response_id=response.id,created=response.created,usage=response.usage.model_dump() if response.usage else None,content=response.choices[0].message.content))
  return response
 eval_fact.client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
 selection=json.loads((OUT/'selection.json').read_text());dest=OUT/'factuality';dest.mkdir(exist_ok=True);tasks=[]
 per={}
 for name in selection['factuality_candidates']:
  path=OUT/('references' if name=='SMFA' else 'candidates')/name/'forget_generation_results.json'
  qs=[q for q in json.loads(path.read_text())['Generation_Questions'] if q['question type'].endswith('memory')];assert len(qs)==325
  per[name]=qs
  atomic_json(dest/f'{name}_input.json',dict(source_sha256=sha256(path),Generation_Questions=qs))
 order=list(range(325));random.Random(20261023).shuffle(order)
 for i in order:
  for name in selection['factuality_candidates']:
   q=per[name][i];tasks.append(dict(model=name,index=i,question=q,digest=hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest()))
 identity=dict(selection_sha256=sha256(OUT/'selection.json'),official_scorer_sha256=sha256(ROOT/'S-MLLMUn/eval_fact.py'),runner_sha256=sha256(Path(__file__)),model='qwen-plus',settings='official defaults; 4 workers; interleaved models, fixed shuffled question order; timeout120, transport retries2, official attempts3; only invalid apostrophe JSON escape repaired',started_at=now())
 atomic_json(dest/'run.json',identity);cache=dest/'scores.jsonl';records={};t=time.monotonic()
 if cache.exists():
  valid={(x['model'],x['index']):x['digest'] for x in tasks}
  for line in cache.read_text().splitlines():
   try:r=json.loads(line)
   except json.JSONDecodeError:continue
   if r.get('digest')==valid.get((r['model'],r['index'])) and r['result'].get('Factuality Score') is not None:records[r['model'],r['index']]=r
 def task(x):
  local.responses=[];tick=time.monotonic();buckets={k:[] for k in ['image_understanding','image_memory','text_memory']};scores={k:[] for k in buckets}
  eval_fact.process_single_question(x['question'],buckets,scores)
  return dict(x,result=buckets[x['question']['question type']][0],api_responses=local.responses,finished_at=now(),seconds=time.monotonic()-tick)
 for attempt in range(1,7):
  pending=[x for x in tasks if (x['model'],x['index']) not in records]
  if not pending:break
  with ThreadPoolExecutor(max_workers=4) as pool,cache.open('a') as stream:
   futures=[pool.submit(task,x) for x in pending]
   for fut in as_completed(futures):
    r=fut.result();r['round']=attempt;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
    if r['result'].get('Factuality Score') is not None:records[r['model'],r['index']]=r
    atomic_json(dest/'status.json',dict(state='scoring',at=now(),valid=len(records),total=975,round=attempt,elapsed_seconds=time.monotonic()-t))
    if len(records)%50==0:print('factuality',len(records),'/975',flush=True)
 if len(records)!=975:raise RuntimeError(f'Missing {975-len(records)} factuality scores after resumable retries')
 result=dict(state='complete',finished_at=now(),seconds=time.monotonic()-t,records=[records[x['model'],x['index']] for x in tasks],run=identity)
 atomic_json(dest/'scores.json',result);atomic_json(dest/'status.json',dict(state='complete',finished_at=now(),valid=975,total=975,seconds=time.monotonic()-t));print('975 judgments complete',flush=True)
if __name__=='__main__':main()
