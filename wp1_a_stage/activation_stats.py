"""Original-model activation moments with bounded GPU memory and S0 eigensystems."""
import sys,json,time,argparse,hashlib,resource,gc
from pathlib import Path
from io import BytesIO
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from transformers import AutoProcessor,LlavaOnevisionForConditionalGeneration
from wp1_a_stage.common import OUT,DATA,rows,now,atomic_json,sha256,frozen
from wp1_review.memory_construct import ORIGINAL,BASE_MODEL,load_lora_matrices
from wp1_a_stage.candidates import correct_a,constructed,group_key

def unit_checks():
 torch.set_num_threads(4);torch.manual_seed(20261023)
 d=3584;r=20;a=torch.randn(16,d);v=torch.linalg.qr(torch.randn(d,r)).Q;e=torch.linspace(10,1,r)
 hard=correct_a(a,v,e);soft0=correct_a(a,v,e,0);softinf=correct_a(a,v,e,1e12)
 residual=float((hard@v).abs().max());comp=torch.randn(d,8);comp=comp-v@(v.T@comp)
 complement=float(((hard-a)@comp).abs().max())
 assert torch.equal(soft0,a) and torch.equal(correct_a(a,v[:,:0],e[:0]),a)
 assert residual<2e-5 and complement<2e-4 and torch.allclose(softinf,hard,atol=1e-6,rtol=1e-6),(residual,complement)
 for lam in [10,100,1000]:assert torch.isfinite(correct_a(a,v,e,lam)).all()
 matrices,scales=load_lora_matrices();stem=next(s.removesuffix('.lora_A.weight') for s in matrices['RA'] if s.endswith('.lora_A.weight') and group_key(s.removesuffix('.lora_A.weight'))=='layer_00_attn')
 A=matrices['MFA_multi'][stem+'.lora_A.weight'];B=matrices['MFA_multi'][stem+'.lora_B.weight'];base=torch.zeros((B.shape[0],A.shape[1]),dtype=torch.float16)
 raw,_=constructed(base,matrices,scales,stem,dict(family='C1',c=1))
 zero,_=constructed(base,matrices,scales,stem,dict(family='C2',lambda_rel=10,alpha=1),{group_key(stem):dict(vectors=torch.empty(d,0),values=torch.empty(0))})
 assert torch.equal(raw,zero)
 guard=json.loads((OUT/'c1_exact_guard.json').read_text());assert guard['state']=='passed'
 result=dict(state='passed',at=now(),dimension=d,hard_protected_max_abs=residual,orthogonal_complement_change_max_abs=complement,lambda0_exact=True,r0_update_exact=True,lambda_large_equals_hard=True,fixed_modules_guard=guard['counts'],finite=True,code_sha256={'candidates.py':sha256(OUT/'candidates.py'),'activation_stats.py':sha256(Path(__file__))})
 atomic_json(OUT/'operator_unit_checks.json',result);return result

def make_inputs(processor,sample,byid):
 image=None
 if sample['type']!='pure_text':
  row=byid[sample['person_id']];data=row['character_image' if sample['type']=='multimodal' else 'ophthalmic_image']['bytes']
  assert hashlib.sha256(data).hexdigest()==sample['image_sha256'];image=Image.open(BytesIO(data)).convert('RGB')
 content=([{'type':'image'}] if image is not None else [])+[{'type':'text','text':sample['question']}]
 chat=[{'role':'user','content':content},{'role':'assistant','content':[{'type':'text','text':sample['answer']}]}]
 prompt=processor.apply_chat_template(chat,add_generation_prompt=False)
 return processor(text=prompt,images=image,return_tensors='pt') if image is not None else processor(text=prompt,return_tensors='pt')

def spectrum(s):
 s=(s+s.T)*.5;e,v=torch.linalg.eigh(s);e=e.flip(0);v=v.flip(1)
 if not torch.isfinite(e).all():raise ValueError('Nonfinite eigenvalue')
 if float(e[-1]) < -1e-4*float(e[0]):raise ValueError('Moment not PSD within FP32 tolerance')
 positive=e.clamp_min(0);r=int(torch.searchsorted(positive.cumsum(0),.9*positive.sum()))+1
 assert 0<r<3584,(r,float(e[0]))
 return dict(vectors=v[:,:r].contiguous().cpu(),values=e[:r].contiguous().cpu(),all_eigenvalues=e.cpu()),r

class Collector:
 def __init__(self,model,layers,prefix):
  self.acc={};self.context=None;self.handles=[];self.prefix=prefix;self.audit=[];self.image_probe={};self.prefix_max_diff=0.;self.visual_probe_max_diff=0.;self.prefix_max_relative_l2=0.;self.visual_probe_max_relative_l2=0.;self.visual_audit=[]
  for layer in layers:
   modules={'attn':model.model.language_model.layers[layer].self_attn.q_proj,'o':model.model.language_model.layers[layer].self_attn.o_proj,'mlp':model.model.language_model.layers[layer].mlp.gate_proj}
   for group,module in modules.items():
    key=f'layer_{layer:02d}_{group}';self.acc[key]={n:torch.zeros(3584,3584,dtype=torch.float32) for n in ['V_unique','V_repeated','T']}
    self.acc[key].update(prefix=None,n_prefix=0,n_v_unique=0,n_v_repeated=0,n_t=0)
    self.handles.append(module.register_forward_pre_hook(self.hook(key)))
 def hook(self,key):
  def fn(module,args):
   x=args[0][0].detach();ctx=self.context;s=self.acc[key];p=x[:len(self.prefix)].float();v=x[ctx['visual']].float();t=x[ctx['text']].float()
   if s['prefix'] is None:s['prefix']=p.cpu()
   else:
    ref=s['prefix'].to(p.device);self.prefix_max_diff=max(self.prefix_max_diff,float((p-ref).abs().max()))
    relative=float(torch.linalg.vector_norm(p-ref)/torch.linalg.vector_norm(ref).clamp_min(1e-12));self.prefix_max_relative_l2=max(self.prefix_max_relative_l2,relative)
    self.audit.append(dict(group=key,max_abs=float((p-ref).abs().max()),relative_l2=relative))
    if relative>.02:
     atomic_json(OUT/'prefix_failure.json',dict(group=key,relative_l2=relative,max_abs=float((p-ref).abs().max()),reference_max_abs=float(ref.abs().max()),audit=self.audit));raise ValueError(f'Prefix activations disagree materially: {key} {relative}')
   s['n_prefix']+=1
   if len(v):
    # v3 section 3.6: one canonical activation per image, repeated by multiplicity.
    s['n_v_repeated']+=len(v)
    if ctx['first_image']:
     vv=(v.T@v).cpu();s['V_unique'].add_(vv);s['V_repeated'].add_(vv,alpha=ctx['multiplicity']);s['n_v_unique']+=len(v)
    probe=v[torch.linspace(0,len(v)-1,min(16,len(v)),device=v.device).long()][:,::113].cpu();pk=(key,ctx['image_sha256'])
    if pk in self.image_probe:
     ref=self.image_probe[pk];diff=float((probe-ref).abs().max());self.visual_probe_max_diff=max(self.visual_probe_max_diff,diff)
     relative=float(torch.linalg.vector_norm(probe-ref)/torch.linalg.vector_norm(ref).clamp_min(1e-12));self.visual_probe_max_relative_l2=max(self.visual_probe_max_relative_l2,relative)
     self.visual_audit.append(dict(group=key,image_sha256=ctx['image_sha256'],max_abs=diff,relative_l2=relative))
     if relative>.01:raise ValueError(f'Visual activation duplicate mismatch materially: {key} relative_l2={relative}')
    else:self.image_probe[pk]=probe
   if len(t):s['T'].add_((t.T@t).cpu());s['n_t']+=len(t)
   assert torch.isfinite(p).all() and torch.isfinite(v).all() and torch.isfinite(t).all()
  return fn
 def close(self):
  for h in self.handles:h.remove()

def collect(model,processor,samples,byid,layers,destination,do_eigen=True):
 destination.mkdir(parents=True,exist_ok=True);prepared=[make_inputs(processor,s,byid) for s in samples];img_id=model.config.image_token_id
 prefix=None
 for inp in prepared:
  ids=inp.input_ids[0];where=(ids==img_id).nonzero()
  if len(where):prefix=ids[:int(where[0])].tolist();break
 assert prefix is not None and len(prefix)>0
 image_prefix=prefix.copy()
 for inp in prepared:
  ids=inp.input_ids[0].tolist();n=0
  while n<min(len(prefix),len(ids)) and prefix[n]==ids[n]:n+=1
  prefix=prefix[:n]
 assert prefix, 'No common prefix: explicit separate handling required'
 for inp in prepared:assert inp.input_ids[0][:len(prefix)].tolist()==prefix
 collector=Collector(model,layers,prefix);seen=set();records=[];torch.cuda.reset_peak_memory_stats();start=time.monotonic();elapsed_previous=0.
 multiplicities=Counter(s['image_sha256'] for s in samples if s['image_sha256'])
 checkpoint=destination/'collection_checkpoint.pt';identity=dict(code_sha256=sha256(Path(__file__)),selection_sha256=sha256(OUT/'b1_selection.json'),layers=layers,samples=samples,prefix=prefix)
 if checkpoint.exists():
  saved=torch.load(checkpoint,map_location='cpu',weights_only=True);assert saved['identity']==identity,'Checkpoint identity changed'
  collector.acc=saved['acc'];collector.image_probe=saved['image_probe'];collector.audit=saved['prefix_audit'];collector.visual_audit=saved['visual_audit'];seen=set(saved['seen']);records=saved['records'];elapsed_previous=saved['elapsed']
  for field in ['prefix_max_diff','prefix_max_relative_l2','visual_probe_max_diff','visual_probe_max_relative_l2']:setattr(collector,field,saved[field])
  del saved;print('Resuming statistics at',len(records),flush=True)
 try:
  for i,(sample,inp) in enumerate(zip(samples,prepared)):
   if i<len(records):continue
   ids=inp.input_ids[0];visual=(ids==img_id);text=~visual;text[:len(prefix)]=False
   assert int(visual.sum())+(len(prefix))+int(text.sum())==len(ids)
   assert bool(visual.any())==(sample['type']!='pure_text')
   imagehash=sample['image_sha256'];first=imagehash is not None and imagehash not in seen
   collector.context=dict(visual=visual.to('cuda'),text=text.to('cuda'),first_image=first,image_sha256=imagehash,multiplicity=multiplicities.get(imagehash,0))
   torch.cuda.synchronize();tick=time.monotonic()
   with torch.inference_mode():
    output=model.model(**inp.to('cuda'),use_cache=False,return_dict=True)
    assert output.last_hidden_state.shape[1]==len(ids)
    del output
   inp.to("cpu")  # BatchFeature.to mutates in place; do not retain all images on GPU.
   torch.cuda.synchronize()
   record=dict(index=i,person_id=sample['person_id'],type=sample['type'],sequence_length=len(ids),prefix_tokens=len(prefix),visual_tokens=int(visual.sum()),text_tokens=int(text.sum()),first_image=first,forward_and_moments_seconds=time.monotonic()-tick)
   records.append(record);seen.add(imagehash)
   atomic_json(destination/'progress.json',dict(state='collecting',at=now(),completed=i+1,total=len(samples),elapsed_seconds=elapsed_previous+time.monotonic()-start,last=record))
   print('statistics',i+1,'/',len(samples),record['sequence_length'],round(record['forward_and_moments_seconds'],2),'s',flush=True)
   if len(samples)>10 and ((i+1)%16==0 or i+1==len(samples)):
    state=dict(identity=identity,acc=collector.acc,image_probe=collector.image_probe,prefix_audit=collector.audit,visual_audit=collector.visual_audit,seen=list(seen),records=records,elapsed=elapsed_previous+time.monotonic()-start)
    for field in ['prefix_max_diff','prefix_max_relative_l2','visual_probe_max_diff','visual_probe_max_relative_l2']:state[field]=getattr(collector,field)
    temporary=checkpoint.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(checkpoint);del state
 finally:collector.close()
 collection_seconds=elapsed_previous+time.monotonic()-start;peak=torch.cuda.max_memory_allocated();groups={};eig_start=time.monotonic();moments=destination/'moments';moments.mkdir(exist_ok=True);eigen=destination/'eigen';eigen.mkdir(exist_ok=True)
 for key,s in collector.acc.items():
  if not all(torch.isfinite(s[k]).all() for k in ['V_unique','V_repeated','T']):raise ValueError('Nonfinite moments')
  torch.save(s,moments/f'{key}.pt');p=s['prefix'].T@s['prefix'];traces={k:float(torch.trace(s[k])) for k in ['V_unique','V_repeated','T']};traces['P']=float(torch.trace(p));denom=traces['V_repeated']+traces['T']+s['n_prefix']*traces['P']
  tick=time.monotonic();spec,r=spectrum((s['V_repeated']+s['T']+s['n_prefix']*p).to('cuda'));torch.cuda.synchronize();sec=time.monotonic()-tick
  torch.save(spec,eigen/f'{key}.pt');groups[key]=dict(r=r,eigen_seconds=sec,visual_energy_fraction_S0=traces['V_repeated']/denom,prefix_energy_fraction_S0=s['n_prefix']*traces['P']/denom,visual_energy_fraction_S0d=traces['V_unique']/(traces['V_unique']+traces['T']+traces['P']),traces=traces,counts={k:s[k] for k in ['n_prefix','n_v_unique','n_v_repeated','n_t']},moments_sha256=sha256(moments/f'{key}.pt'),eigen_sha256=sha256(eigen/f'{key}.pt'))
  print('eigen',key,'r',r,round(sec,2),'s',flush=True)
 summary=dict(state='complete',at=now(),samples=len(samples),layers=layers,prefix_token_ids=prefix,image_prefix_token_ids=image_prefix,nonshared_prefix_policy='Leading tokens beyond the common prefix enter T',prefix_activation_max_abs_difference=collector.prefix_max_diff,prefix_activation_max_relative_l2=collector.prefix_max_relative_l2,prefix_tolerance_note="FP16 sequence-length kernel variation: relative Frobenius difference must stay <=2%; actual differences retained",prefix_checks=collector.audit,visual_duplicate_probe_max_abs_difference=collector.visual_probe_max_diff,visual_duplicate_probe_max_relative_l2=collector.visual_probe_max_relative_l2,visual_checks=collector.visual_audit,duplicate_check='16 evenly spaced visual tokens, every 113th feature per group; <=1% relative Frobenius difference. Canonical first-image moments weighted by exact multiplicity for S0, per v3 3.6; S0d uses weight1.',accumulator_location='CPU; each covariance product computed FP32 on GPU; TF32 disabled',matrix_bytes=3584*3584*4,accumulator_bytes=len(layers)*3*3*3584*3584*4,peak_gpu_allocated_bytes=peak,peak_cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,collection_seconds=collection_seconds,eigendecomposition_and_save_seconds=time.monotonic()-eig_start,groups=groups,sample_records=records,source_selection_sha256=sha256(OUT/'b1_selection.json'),code_sha256=sha256(Path(__file__)))
 atomic_json(destination/'S0_summary.json',summary);return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--unit',action='store_true');p.add_argument('--pilot-only',action='store_true');args=p.parse_args();frozen()
 if args.unit:print(unit_checks());return
 assert json.loads((OUT/'operator_unit_checks.json').read_text())['state']=='passed'
 torch.set_num_threads(4);torch.manual_seed(20261023);torch.backends.cuda.matmul.allow_tf32=False
 samples=json.loads((OUT/'b1_selection.json').read_text())['selected'];frame=rows(DATA/'unlearning_data/retain_few_shot_5/train-00000-of-00001.parquet');byid={row.ID:row for _,row in frame.iterrows()}
 model=LlavaOnevisionForConditionalGeneration.from_pretrained(str(ORIGINAL),torch_dtype=torch.float16,device_map={'':'cuda:0'},low_cpu_mem_usage=True,local_files_only=True).eval();processor=AutoProcessor.from_pretrained(str(BASE_MODEL),local_files_only=True)
 # First sample of each type, then frozen order to ten. No outcome-based choices.
 idx=[]
 for kind in ['multimodal','ophthalmic','pure_text']:idx.append(next(i for i,s in enumerate(samples) if s['type']==kind))
 idx+= [i for i in range(len(samples)) if i not in idx][:10-len(idx)];pilot=[samples[i] for i in idx]
 summary=collect(model,processor,pilot,byid,[0,27],OUT/'pilot')
 estimate=dict(at=now(),pilot_indices=idx,full_collection_seconds_linear_layer_bound=summary['collection_seconds']*(128/10)*14,full_eigen_seconds=sum(v['eigen_seconds'] for v in summary['groups'].values())*14,full_cpu_accumulator_bytes=84*3*3584*3584*4,full_moment_disk_bytes=84*3*3584*3584*4,pilot_peak_gpu_bytes=summary['peak_gpu_allocated_bytes'],note='Layer-linear bound overestimates model forward cost, which is paid once. CPU accumulators keep GPU memory roughly independent of collected layer count; concurrent evaluation may slow both jobs.')
 atomic_json(OUT/'pilot/cost_estimate.json',estimate);print('pilot passed',estimate,flush=True)
 if not args.pilot_only:
  gc.collect();torch.cuda.empty_cache();collect(model,processor,samples,byid,list(range(28)),OUT/'statistics')
if __name__=='__main__':main()
