"""Frozen F1 appearance positive-control test; at most one whole-bank switch."""
import os,sys,json,time,gc
from pathlib import Path
from io import BytesIO
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,LlavaOnevisionForConditionalGeneration
from rouge_score.rouge_scorer import RougeScorer
from wp1_review.memory_construct import ORIGINAL,SCULPTED,BASE_MODEL,REPO,official_answer
from wp1_a_stage.common import dev_frames,refusal_phrases,normalize,LOOSE_REFUSAL
from wp1_supplement.common import OUT,now,atomic_json,sha256,frozen,after_freeze

def refused(answer):
    exact=normalize(answer) in refusal_phrases()
    return dict(refusal_exact=float(exact),refusal_loose=float(exact or bool(LOOSE_REFUSAL.search(answer))))

def assess(bank):
    dest=OUT/'appearance'/bank
    answers={name:json.loads((dest/name/'generation_results.json').read_text())['questions'] for name in ('Original','SMFA','MFA_unsculpted')}
    scorer=RougeScorer(['rougeL'],use_stemmer=True)
    original=answers['Original'];reports={};details={}
    people=sorted({q['person_id'] for q in original})
    for name in ('SMFA','MFA_unsculpted'):
        pairs=[]
        for a,b in zip(original,answers[name]):
            assert all(a[k]==b[k] for k in ('person_id','question','attribute'))
            ra,rb=refused(a['generated_answer']),refused(b['generated_answer'])
            pairs.append(dict(person_id=a['person_id'],attribute=a['attribute'],
                              agreement=scorer.score(a['generated_answer'],b['generated_answer'])['rougeL'].fmeasure,
                              original_refusal=ra['refusal_loose'],model_refusal=rb['refusal_loose'],
                              refusal_increment=rb['refusal_loose']-ra['refusal_loose']))
        agreement=float(np.mean([p['agreement'] for p in pairs]));increment=float(np.mean([p['refusal_increment'] for p in pairs]))
        by_person=defaultdict(list)
        for p in pairs:by_person[p['person_id']].append(p)
        assert len(pairs)==150 and len(by_person)==25 and all(len(v)==6 for v in by_person.values())
        arrays={key:np.array([np.mean([p[key] for p in by_person[pid]]) for pid in people]) for key in ('agreement','refusal_increment')}
        draws=np.random.default_rng(20261023).integers(0,25,(20000,25))
        ci={key:np.quantile(values[draws].mean(1),[.025,.975]).tolist() for key,values in arrays.items()}
        subsets={}
        for label,ref in [('original_nonrefusal',0),('original_refusal',1)]:
            subset=[p for p in pairs if p['original_refusal']==ref]
            subsets[label]=dict(n=len(subset),agreement=float(np.mean([p['agreement'] for p in subset])) if subset else None,
                                refusal_increment=float(np.mean([p['refusal_increment'] for p in subset])) if subset else None)
        reports[name]=dict(mean_agreement_rougeL=agreement,refusal_increment=increment,
                           original_refusal_rate=float(np.mean([p['original_refusal'] for p in pairs])),
                           model_refusal_rate=float(np.mean([p['model_refusal'] for p in pairs])),
                           insensitive=bool(increment<.02 and agreement>=.90),conditional_ci95_by_person=ci,subsets=subsets,
                           by_attribute={a:dict(n=sum(p['attribute']==a for p in pairs),
                                                agreement=float(np.mean([p['agreement'] for p in pairs if p['attribute']==a])),
                                                refusal_increment=float(np.mean([p['refusal_increment'] for p in pairs if p['attribute']==a]))) for a in sorted({p['attribute'] for p in pairs})})
        details[name]=pairs
    report=dict(state='complete',at=now(),bank=bank,n_people=25,n_prompts=150,
                positive_control='MFA_unsculpted versus Original',comparisons=reports,
                insensitive=reports['MFA_unsculpted']['insensitive'],source_sha256={name:sha256(dest/name/'generation_results.json') for name in answers},
                ci_note='Paired person bootstrap, 20000 draws, seed 20261023; diagnostic only, not template selection by CI.')
    atomic_json(dest/'paired_details.json',details);atomic_json(dest/'sensitivity.json',report)
    return report

def run_bank(bank,processor,frame):
    templates=json.loads((OUT/'templates.json').read_text())[bank]
    paths={'Original':ORIGINAL,'SMFA':SCULPTED,'MFA_unsculpted':REPO/'model/unlearned_model/llava-onevision/split_ratio_5/MFA_unsculpted/sculpted'}
    for name,path in paths.items():
        dest=OUT/'appearance'/bank/name;dest.mkdir(parents=True,exist_ok=True)
        identity=dict(bank=bank,model=name,checkpoint=str(path),checkpoint_index_sha256=sha256(path/'model.safetensors.index.json'),
                      templates_sha256=sha256(OUT/'templates.json'),freeze_sha256=sha256(OUT/'FREEZE_supplement.sha256'),code_sha256=sha256(Path(__file__)))
        manifest=dest/'manifest.json'
        if manifest.exists():assert json.loads(manifest.read_text())==identity,'Appearance cache identity changed'
        else:atomic_json(manifest,identity)
        if (dest/'generation_results.json').exists():continue
        status=dict(state='loading',started_at=now(),pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),model=name,bank=bank,completed=0,total=150)
        after_freeze(status['started_at']);atomic_json(dest/'status.json',status)
        start=time.monotonic()
        model=LlavaOnevisionForConditionalGeneration.from_pretrained(str(path),torch_dtype=torch.float16,device_map={'':'cuda:0'},low_cpu_mem_usage=True,local_files_only=True).eval()
        status['load_seconds']=time.monotonic()-start;start=time.monotonic();allq=[]
        status.update(state='generating',generation_started_at=now());after_freeze(status['generation_started_at'])
        for _,row in frame.iterrows():
            cache=dest/'people'/f'{row.ID}.json'
            if cache.exists():
                qs=json.loads(cache.read_text())['questions']
                assert len(qs)==6
                for q,t in zip(qs,templates):assert q['person_id']==row.ID and all(q[k]==v for k,v in t.items())
            else:
                image=Image.open(BytesIO(row.character_image['bytes']));qs=[]
                for template in templates:
                    tick=time.monotonic();answer=official_answer(processor,model,template['question'],image)
                    qs.append(dict(template,person_id=row.ID,cohort='F1_development',bank=bank,model=name,
                                   generated_answer=answer,generated_at=now(),generation_seconds=time.monotonic()-tick,
                                   answer_tokens=len(processor.tokenizer.encode(answer,add_special_tokens=False)),**refused(answer)))
                image.close();atomic_json(cache,dict(questions=qs))
            allq+=qs
            status.update(completed=len(allq),updated_at=now(),elapsed_generation_seconds=time.monotonic()-start)
            atomic_json(dest/'status.json',status);print(bank,name,len(allq),'/150',flush=True)
        assert len(allq)==150
        atomic_json(dest/'generation_results.json',dict(questions=allq))
        status.update(state='complete',finished_at=now(),generation_seconds=time.monotonic()-start);atomic_json(dest/'status.json',status)
        del model;gc.collect();torch.cuda.empty_cache()
    return assess(bank)

def main():
    frozen();torch.set_num_threads(4);torch.manual_seed(20261023);torch.backends.cuda.matmul.allow_tf32=False
    frame=dev_frames()['forget'];processor=AutoProcessor.from_pretrained(str(BASE_MODEL),local_files_only=True)
    primary=run_bank('appearance_holdout',processor,frame)
    active='appearance_holdout'
    if primary['insensitive']:
        revision=OUT/'appearance/protocol_revision.json'
        if not revision.exists():
            atomic_json(revision,dict(at=now(),reason='Prespecified primary bank failed MFA positive-control sensitivity.',
                        rule='Refusal increment <0.02 AND mean agreement >=0.90; switch entire bank once.',
                        old_bank='appearance_holdout',new_bank='appearance_holdout_backup',
                        primary_result_sha256=sha256(OUT/'appearance/appearance_holdout/sensitivity.json'),
                        frozen_template_sha256=sha256(OUT/'templates.json'),no_questionwise_selection=True))
        run_bank('appearance_holdout_backup',processor,frame);active='appearance_holdout_backup'
    report=json.loads((OUT/'appearance'/active/'sensitivity.json').read_text())
    atomic_json(OUT/'appearance/summary.json',dict(state='complete',at=now(),active_bank=active,primary_insensitive=primary['insensitive'],
                active_insensitive=report['insensitive'],further_template_changes_allowed=False,
                limitation='Agreement is relative to Original, not ground-truth visual accuracy. No further bank search even if backup is insensitive.'))

if __name__=='__main__':main()
