"""Nonselective prompt-only coverage/LoRA-response diagnostics, all 28 layers."""
import os,sys,json,time,hashlib,gc,resource
from pathlib import Path
from io import BytesIO
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from transformers import AutoProcessor,LlavaOnevisionForConditionalGeneration
from wp1_a_stage.common import DATA,rows
from wp1_a_stage.candidates import group_key,correct_a
from wp1_a_stage.activation_stats import spectrum
from wp1_review.memory_construct import ORIGINAL,BASE_MODEL,load_lora_matrices,layer_stems
from wp1_supplement.common import OUT,PARENT,now,atomic_json,sha256,frozen,after_freeze

DEST=OUT/'diagnostics'
TOKENS=('V_unique','T_image','T_pure')

def save_pt(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    torch.save(value,temp);temp.replace(path)

def unit_checks():
    torch.manual_seed(20261023)
    x=torch.randn(13,11,dtype=torch.float64)
    a=torch.randn(4,11,dtype=torch.float64);b=torch.randn(9,4,dtype=torch.float64)
    covariance=x.T@x;gram=b.T@b
    lowrank=torch.trace(gram@(a@covariance@a.T))
    explicit=((x@(b@a).T)**2).sum()
    v=torch.linalg.qr(torch.randn(11,3,dtype=torch.float64)).Q
    projection=(x@v).square().sum()/x.square().sum()
    via_trace=torch.trace(v.T@covariance@v)/torch.trace(covariance)
    assert torch.allclose(lowrank,explicit,rtol=1e-12,atol=1e-10)
    assert torch.allclose(projection,via_trace,rtol=1e-12,atol=1e-12)
    assert 0<=projection<=1
    atomic_json(DEST/'unit_checks.json',dict(state='passed',at=now(),lowrank_energy_abs_error=float(abs(lowrank-explicit)),
                coverage_abs_error=float(abs(projection-via_trace)),code_sha256=sha256(Path(__file__))))

def inputs_for(processor,q,byid):
    image=None;imagehash=None
    if q['question type']!='text_memory':
        row=byid[(q['split'],q['person_id'])]
        blob=row['character_image' if q['image_type']=='character' else 'ophthalmic_image']['bytes']
        imagehash=hashlib.sha256(blob).hexdigest();image=Image.open(BytesIO(blob)).convert('RGB')
    content=([dict(type='image')] if image is not None else [])+[dict(type='text',text=q['question'])]
    chat=[dict(role='user',content=content)]
    prompt=processor.apply_chat_template(chat,add_generation_prompt=True)
    inputs=processor(text=prompt,images=image,return_tensors='pt') if image is not None else processor(text=prompt,return_tensors='pt')
    if image is not None:image.close()
    return inputs,imagehash

class Collector:
    def __init__(self,model,layers,prefix):
        self.acc={};self.handles=[];self.context=None;self.prefix=prefix
        self.probes={};self.prefix_diff=0.;self.visual_diff=0.
        for layer in layers:
            modules=dict(attn=model.model.language_model.layers[layer].self_attn.q_proj,
                         o=model.model.language_model.layers[layer].self_attn.o_proj,
                         mlp=model.model.language_model.layers[layer].mlp.gate_proj)
            for group,module in modules.items():
                key=f'layer_{layer:02d}_{group}'
                self.acc[key]={token:torch.zeros(3584,3584,dtype=torch.float32) for token in TOKENS}
                self.acc[key].update(prefix=None,n_V_unique=0,n_V_repeated=0,n_T_image=0,n_T_pure=0)
                self.handles.append(module.register_forward_pre_hook(self.hook(key)))

    def hook(self,key):
        def fn(module,args):
            x=args[0][0].detach();ctx=self.context;s=self.acc[key]
            assert len(x)==ctx['length']
            p=x[:len(self.prefix)].float()
            if s['prefix'] is None:s['prefix']=p.cpu()
            else:
                ref=s['prefix'].to(p.device)
                rel=float(torch.linalg.vector_norm(p-ref)/torch.linalg.vector_norm(ref).clamp_min(1e-12))
                self.prefix_diff=max(self.prefix_diff,rel)
                assert rel<=.02,('Prefix variation',key,rel)
            if ctx['has_image']:
                visual=x[ctx['visual']].float();s['n_V_repeated']+=len(visual)
                probe=visual[torch.linspace(0,len(visual)-1,min(16,len(visual)),device=visual.device).long()][:,::113].cpu()
                probe_key=(key,ctx['imagehash'])
                if probe_key not in self.probes:self.probes[probe_key]=probe
                else:
                    ref=self.probes[probe_key]
                    rel=float(torch.linalg.vector_norm(probe-ref)/torch.linalg.vector_norm(ref).clamp_min(1e-12))
                    self.visual_diff=max(self.visual_diff,rel)
                    assert rel<=.01,('Visual variation',key,rel)
                if ctx['first_image']:
                    s['V_unique'].add_((visual.T@visual).cpu());s['n_V_unique']+=len(visual)
            text=x[ctx['text']].float()
            if len(text):
                token='T_image' if ctx['has_image'] else 'T_pure'
                s[token].add_((text.T@text).cpu());s['n_'+token]+=len(text)
            assert torch.isfinite(p).all() and torch.isfinite(text).all()
        return fn

    def close(self):
        for h in self.handles:h.remove()

def collect(model,processor,samples,byid,layers,dest):
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists():return json.loads((dest/'summary.json').read_text())
    prefix=json.loads((PARENT/'statistics/S0_summary.json').read_text())['prefix_token_ids']
    assert prefix==[151644,872]
    image_token=model.config.image_token_id
    identity=dict(code_sha256=sha256(Path(__file__)),selection_sha256=sha256(OUT/'diagnostic_selection.json'),
                  freeze_sha256=sha256(OUT/'FREEZE_supplement.sha256'),samples=samples,layers=layers,prefix=prefix)
    collector=Collector(model,layers,prefix);seen=set();records=[];previous=0.;started=now()
    after_freeze(started)
    checkpoint=dest/'checkpoint.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['identity']==identity,'Diagnostic cache identity changed'
        collector.acc=saved['acc'];collector.probes=saved['probes'];collector.prefix_diff=saved['prefix_diff'];collector.visual_diff=saved['visual_diff']
        seen=set(saved['seen']);records=saved['records'];previous=saved['seconds'];started=saved['started_at'];del saved
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        for index,q in enumerate(samples):
            if index<len(records):continue
            inp,imagehash=inputs_for(processor,q,byid);ids=inp.input_ids[0]
            assert ids[:len(prefix)].tolist()==prefix
            visual=ids==image_token;text=~visual;text[:len(prefix)]=False
            has_image=q['question type']!='text_memory'
            assert bool(visual.any())==has_image
            assert int(visual.sum()+text.sum())+len(prefix)==len(ids)
            collector.context=dict(visual=visual.to('cuda'),text=text.to('cuda'),length=len(ids),
                                   has_image=has_image,first_image=imagehash not in seen,imagehash=imagehash)
            tick=time.monotonic()
            with torch.inference_mode():
                result=model.model(**inp.to('cuda'),use_cache=False,return_dict=True)
                assert result.last_hidden_state.shape[1]==len(ids)
                del result
            inp.to('cpu');torch.cuda.synchronize()
            record=dict(index=index,person_id=q['person_id'],question_type=q['question type'],image_type=q.get('image_type'),
                        sequence_tokens=len(ids),visual_tokens=int(visual.sum()),text_tokens=int(text.sum()),
                        first_image=imagehash is not None and imagehash not in seen,seconds=time.monotonic()-tick)
            if imagehash is not None:seen.add(imagehash)
            records.append(record)
            atomic_json(dest/'progress.json',dict(state='collecting',started_at=started,at=now(),completed=index+1,total=len(samples),
                        seconds=previous+time.monotonic()-start,last=record,prompt_only=True))
            print(dest.name,index+1,'/',len(samples),round(record['seconds'],2),'s',flush=True)
            if len(samples)>10 and ((index+1)%50==0 or index+1==len(samples)):
                save_pt(checkpoint,dict(identity=identity,acc=collector.acc,probes=collector.probes,prefix_diff=collector.prefix_diff,
                                       visual_diff=collector.visual_diff,seen=list(seen),records=records,
                                       seconds=previous+time.monotonic()-start,started_at=started))
            del inp
    finally:collector.close()
    groups={}
    for key,s in collector.acc.items():
        assert all(torch.isfinite(s[token]).all() for token in TOKENS)
        path=dest/'moments'/f'{key}.pt';save_pt(path,s)
        groups[key]=dict(counts={k:v for k,v in s.items() if k.startswith('n_')},
                         traces={token:float(torch.trace(s[token])) for token in TOKENS},sha256=sha256(path))
    summary=dict(state='complete',started_at=started,finished_at=now(),samples=len(samples),layers=layers,
                 prefix_token_ids=prefix,prefix_excluded=True,prompt_only=True,no_answers_or_generation=True,
                 visual_duplicate_relative_max=collector.visual_diff,prefix_relative_max=collector.prefix_diff,
                 seconds=previous+time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                 cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,groups=groups,records=records,
                 identity=identity)
    atomic_json(dest/'summary.json',summary)
    del collector;gc.collect();torch.cuda.empty_cache()
    return summary

def spectral_counts():
    groups={}
    for path in sorted((PARENT/'statistics/eigen').glob('*.pt')):
        spec=torch.load(path,map_location='cpu',weights_only=True);e=spec['all_eigenvalues'];r=len(spec['values'])
        groups[path.stem]=dict(r=r,counts={str(t):dict(full=int((e/e[0]>t).sum()),protected=int((e[:r]/e[0]>t).sum())) for t in (.1,.01,.001)},
                               source_sha256=sha256(path))
    atomic_json(DEST/'spectral_threshold_counts.json',dict(at=now(),groups=groups,thresholds=[.1,.01,.001],strict_greater=True))

def derived_spectra(layers):
    for layer in layers:
        for group in ('attn','o','mlp'):
            key=f'layer_{layer:02d}_{group}'
            needed=[name for name in ('S0d','S1') if not (DEST/'eigen'/name/f'{key}.pt').exists()]
            if not needed:continue
            s=torch.load(PARENT/'statistics/moments'/f'{key}.pt',map_location='cpu',weights_only=True)
            prefix=s['prefix'].T@s['prefix']
            for name in needed:
                if name=='S0d':matrix=s['V_unique']+s['T']+prefix
                else:matrix=.5*s['V_unique']/s['n_v_unique']+.5*(s['T']+prefix)/(s['n_t']+len(s['prefix']))
                spec,r=spectrum(matrix.to('cuda'));torch.cuda.synchronize()
                save_pt(DEST/'eigen'/name/f'{key}.pt',spec)
                print('diagnostic eigen',name,key,'r',r,flush=True)
                del matrix,spec
            del s,prefix;gc.collect()

def response(a,bgram,c):
    small=a@c@a.T
    value=float(torch.trace(bgram.double()@small.double()))
    if value<0 and abs(value)<1e-5*float((bgram.double()*small.double()).abs().sum()):return 0.
    assert value>=0,('negative LoRA response energy',value)
    return value

def analyze(layers):
    matrices,scales=load_lora_matrices();coverage=[];responses=[]
    for layer in layers:
        for group in ('attn','o','mlp'):
            key=f'layer_{layer:02d}_{group}'
            spectra={'S0':torch.load(PARENT/'statistics/eigen'/f'{key}.pt',map_location='cuda',weights_only=True)}
            for name in ('S0d','S1'):spectra[name]=torch.load(DEST/'eigen'/name/f'{key}.pt',map_location='cuda',weights_only=True)
            stems=[stem for stem in layer_stems(matrices) if group_key(stem)==key]
            factors={}
            for stem in stems:
                a=torch.cat([matrices[name][stem+'.lora_A.weight'] for name in ('MFA_multi','MFA_text')]).to('cuda')
                b=torch.cat([.5*scales[name]*matrices[name][stem+'.lora_B.weight'] for name in ('MFA_multi','MFA_text')],dim=1).to('cuda')
                gram=b.T@b;spec=spectra['S0']
                corrected={f'C2_l{lam}':correct_a(a,spec['vectors'],spec['values'],lam) for lam in (10,100,1000)}
                corrected['C3_hard']=correct_a(a,spec['vectors'],spec['values'])
                factors[stem]=(a,gram,corrected)
            for dataset in ('D_fm','D_rh','D_fi'):
                moments=torch.load(DEST/dataset/'moments'/f'{key}.pt',map_location='cpu',weights_only=True)
                for token in ('V_unique','T_image','T_pure','T_all'):
                    matrix=(moments['T_image']+moments['T_pure']) if token=='T_all' else moments[token]
                    n=(moments['n_T_image']+moments['n_T_pure']) if token=='T_all' else moments['n_'+token]
                    trace=float(torch.trace(matrix));matrix=matrix.to('cuda')
                    for specname,spec in spectra.items():
                        cov=float(((matrix@spec['vectors'])*spec['vectors']).sum())/trace if trace>0 else None
                        assert cov is None or -.00001<=cov<=1.00001,(key,specname,cov)
                        coverage.append(dict(dataset=dataset,group=key,token_type=token,n_tokens=n,spectrum=specname,
                                             r=len(spec['values']),coverage=cov,raw_trace=trace))
                    for stem,(a,gram,corrected) in factors.items():
                        raw=response(a,gram,matrix) if trace>0 else 0.
                        for candidate,ac in corrected.items():
                            corrected_energy=response(ac,gram,matrix) if trace>0 else 0.
                            responses.append(dict(dataset=dataset,group=key,module=stem.split('.')[-1],token_type=token,
                                                  n_tokens=n,candidate=candidate,raw_energy=raw,corrected_energy=corrected_energy,
                                                  response_ratio=(corrected_energy/raw)**.5 if raw>0 else None))
                    del matrix
                del moments
            del factors,spectra;gc.collect();torch.cuda.empty_cache()
            atomic_json(DEST/'analysis_progress.json',dict(at=now(),last_group=key,coverage_rows=len(coverage),response_rows=len(responses)))
    atomic_json(DEST/'coverage.json',dict(state='complete',at=now(),rows=coverage,
                definition='tr(Vr^T C Vr)/tr(C); Original prompt inputs, prefix excluded; visual unique by image hash within dataset; S0/S0d/S1 protected tau=.9.',
                source_selection_sha256=sha256(OUT/'diagnostic_selection.json')))
    atomic_json(DEST/'response_ratios.json',dict(state='complete',at=now(),rows=responses,
                definition='sqrt(tr(dW_corrected C dW_corrected^T)/tr(dW_raw C dW_raw^T)); averaged MFA_multi+MFA_text update, alpha=1, FP32 factors; excludes base-weight rounding.',
                aggregation_note='Per module. Group aggregation, if used, is sqrt(sum corrected energy/sum raw energy), not average ratios.',
                undefined_policy='Zero input trace or zero raw response returns null, not zero ratio.'))

def main():
    frozen();DEST.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(20261023);torch.backends.cuda.matmul.allow_tf32=False
    atomic_json(DEST/'status.json',dict(state='starting',started_at=now(),pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    unit_checks();spectral_counts()
    selection=json.loads((OUT/'diagnostic_selection.json').read_text());primary_layers=selection['layers'];layers=list(range(28))
    scope=DEST/'scope_extension.json'
    if not scope.exists():
        atomic_json(scope,dict(at=now(),frozen_primary_layers=primary_layers,additional_layers=[l for l in layers if l not in primary_layers],
                    reason='Complete the requested per-layer diagnostic; deterministic complement, no result-based layer choice.',
                    selection_rules_unchanged=True,diagnostics_do_not_affect_selection=True,source_selection_sha256=sha256(OUT/'diagnostic_selection.json')))
    byid={}
    for split in ('forget','retain'):
        frame=rows(DATA/f'eval_data/{split}_5/train-00000-of-00001.parquet')
        byid.update({(split,row.ID):row for _,row in frame.iterrows()})
    model=LlavaOnevisionForConditionalGeneration.from_pretrained(str(ORIGINAL),torch_dtype=torch.float16,device_map={'':'cuda:0'},low_cpu_mem_usage=True,local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(str(BASE_MODEL),local_files_only=True)
    samples=selection['selected']['D_fm']
    indices=[]
    for kind in ('character','ophthalmic',None):
        indices.append(next(i for i,q in enumerate(samples) if (q['image_type'] if q['question type']!='text_memory' else None)==kind))
    indices+=[i for i in range(len(samples)) if i not in indices][:10-len(indices)]
    pilot=collect(model,processor,[samples[i] for i in indices],byid,[0,27],DEST/'pilot')
    atomic_json(DEST/'pilot/validation.json',dict(state='passed',at=now(),indices=indices,samples=10,layers=2,
                prompt_only=True,conservation='Visual + text + common prefix equals sequence length for every sample.',
                prefix_max_relative=pilot['prefix_relative_max'],visual_duplicate_max_relative=pilot['visual_duplicate_relative_max']))
    for dataset in ('D_fm','D_rh','D_fi'):
        collect(model,processor,selection['selected'][dataset],byid,layers,DEST/dataset)
    del model,processor,byid;gc.collect();torch.cuda.empty_cache()
    derived_spectra(layers);analyze(layers)
    atomic_json(DEST/'status.json',dict(state='complete',finished_at=now(),layers=layers,frozen_primary_layers=primary_layers,unit_checks='passed',pilot='passed',
                sampling_note='All 28 layers; frozen ten-layer primary slice retained, other layers added before diagnostic collection solely for complete reporting.',
                optional_half_B1='Not run; optional and not needed for B1 selection.',code_sha256=sha256(Path(__file__))))

if __name__=='__main__':main()
