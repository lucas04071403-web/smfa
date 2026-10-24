"""Shared A-stage data boundary, metrics and immutable-input verification."""
import sys,json,hashlib,re
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wp1_review.prepare_wp1 import DATA,EVAL,rows,annotate_questions,atomic_json,sha256,refusal_phrases,normalize,LOOSE_REFUSAL
from rouge_score.rouge_scorer import RougeScorer
OUT=ROOT/'wp1_a_stage'
def now():return datetime.now(timezone.utc).isoformat()
def frozen():
 for line in (OUT/'FREEZE.sha256').read_text().splitlines():
  digest,name=line.split('  ',1)
  if sha256(OUT/name)!=digest:raise ValueError(f'Frozen file changed: {name}')
 p=json.loads((ROOT/'wp1_review/fixed_person_split.json').read_text())
 if sha256(ROOT/'wp1_review/fixed_person_split.json')!=json.loads((OUT/'protocol.json').read_text())['split_sha256']:raise ValueError('Split changed')
 return p

def dev_frames():
 part=frozen();result={}
 for split,key in [('forget','F1_development'),('retain','development')]:
  f=rows(DATA/f'eval_data/{split}_5/train-00000-of-00001.parquet')
  result[split]=f[f.ID.isin(part[split][key])]
 assert len(result['forget'])==25 and len(result['retain'])==100
 return result

def questions_for_row(row,split,source_person_index):
 result=[]
 for kind,column in [('image_understanding','image understanding'),('image_memory','image memory'),('text_memory','text memory')]:
  for q in row[column]:
   result.append({'question type':kind,'image_id':None if kind=='text_memory' else row.ID,'image_type':q.get('Image'),'question':q['Question'],'ground_truth':q['Answer'],'person_id':row.ID,'cohort':'F1_development' if split=='forget' else 'retain_development','split':split,'source_person_index':int(source_person_index),'person_question_index':len(result)})
 return result

def metrics(questions):
 scorer=RougeScorer(['rougeL'],use_stemmer=True);phrases=refusal_phrases();buckets=defaultdict(list)
 for q in questions:
  answer=q['generated_answer'];exact=normalize(answer) in phrases
  vals={'rougeL':scorer.score(q['ground_truth'],answer)['rougeL'].fmeasure,'refusal_exact':float(exact),'refusal_loose':float(exact or bool(LOOSE_REFUSAL.search(answer))),'answer_words':len(answer.split()),'answer_characters':len(answer)}
  if 'answer_tokens' in q:vals['answer_tokens']=q['answer_tokens']
  kind=q['question type'];labels=[kind,'all']
  if kind.endswith('memory'):labels.append('combined_memory')
  if q.get('image_type'):labels.append(kind+'/'+q['image_type'])
  for label in labels:buckets[q['split']+'/'+label].append((q['person_id'],vals))
 result={}
 for label,items in sorted(buckets.items()):
  per=defaultdict(list)
  for pid,v in items:per[pid].append(v)
  keys=set.intersection(*(set(v) for _,v in items))
  result[label]={'n_questions':len(items),'n_people':len(per),'mean':{k:sum(v[k] for _,v in items)/len(items) for k in sorted(keys)},'person_macro_mean':{k:sum(sum(v[k] for v in vs)/len(vs) for vs in per.values())/len(per) for k in sorted(keys)}}
 return result

def references():
 frames=dev_frames();part=frozen()
 for model in ['Original','Base','SMFA','MFA_unsculpted']:
  out=OUT/'references'/model;out.mkdir(parents=True,exist_ok=True);allq=[];provenance={}
  for split,key in [('forget','F1_development'),('retain','development')]:
   path=EVAL/model/f'{split}_generation_results.json';raw=json.loads(path.read_text())['Generation_Questions']
   full=rows(DATA/f'eval_data/{split}_5/train-00000-of-00001.parquet')
   qs=[dict(q,person_id=pid,split=split,cohort=key,source_generation_index=i) for i,(q,pid) in enumerate(annotate_questions(raw,full)) if pid in part[split][key]]
   atomic_json(out/f'{split}_generation_results.json',{'Generation_Questions':qs});allq+=qs;provenance[split]=sha256(path)
  assert len(allq)==2375
  atomic_json(out/'metrics.json',metrics(allq));atomic_json(out/'provenance.json',dict(source_sha256=provenance,created_at=now(),split_sha256=sha256(ROOT/'wp1_review/fixed_person_split.json')))
if __name__=='__main__':references()
