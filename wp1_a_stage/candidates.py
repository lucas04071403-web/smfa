"""FP32 LoRA corrections, official FP16 base arithmetic, and exact model guards."""
import sys,json,re,time
from pathlib import Path
from contextlib import ExitStack
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wp1_review.memory_construct import load_lora_matrices,layer_stems,construct_weight,ORIGINAL,SCULPTED,REPO,checkpoint_names
from wp1_a_stage.common import OUT,now,atomic_json,frozen,sha256
INITIAL=[dict(name='C0_k1',family='C0',k_multi=1,k_text=2),dict(name='C0_k10',family='C0',k_multi=10,k_text=20)]+[dict(name='C1_c'+str(c).replace('.','p'),family='C1',c=c) for c in [.25,.5,.75,1.0]]
SPECTRAL=[dict(name=f'C2_l{v}',family='C2',lambda_rel=v,alpha=1) for v in [10,100,1000]]+[dict(name='C3_hard',family='C3',alpha=1)]

def group_key(stem):
 m=re.search(r'model\.language_model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)$',stem)
 if not m:return None
 layer,block,module=m.groups()
 group='attn' if module in ['q_proj','k_proj','v_proj'] else 'o' if module=='o_proj' else 'mlp' if module in ['gate_proj','up_proj'] else None
 return f'layer_{int(layer):02d}_{group}' if group else None

def correct_a(a,v,e,lam=None):
 if v.shape[1]==0 or lam==0:return a.clone()
 av=a@v
 if lam is not None:
  w=(lam*(e/e[0]))/(1+lam*(e/e[0]));av=av*w
 return a-av@v.T

def constructed(base,matrices,scales,stem,config,eigens=None):
 group=group_key(stem)
 if config['family']=='C0' or group is None:
  return construct_weight(base,matrices,scales,stem,config.get('k_multi',5),config.get('k_text',10))
 delta=[]
 for name in ['MFA_multi','MFA_text']:
  a=matrices[name][stem+'.lora_A.weight'];b=matrices[name][stem+'.lora_B.weight']
  if config['family'] in ['C2','C3']:
   spec=eigens[group];a=correct_a(a,spec['vectors'],spec['values'],config.get('lambda_rel'))
  delta.append(scales[name]*(b@a))
 update=(delta[0]+delta[1])/2
 update=update*config.get('c',config.get('alpha',1))
 return (base.half()+update).half(),{'MFA_multi':0,'MFA_text':0}

def iter_weights(matrices,scales,config,eigens=None):
 index=json.loads((ORIGINAL/'model.safetensors.index.json').read_text())['weight_map']
 with ExitStack() as stack:
  handles={s:stack.enter_context(safe_open(ORIGINAL/s,framework='pt',device='cpu')) for s in set(index.values())}
  for stem in layer_stems(matrices):
   runtime=stem.removeprefix('base_model.model.')+'.weight';key=checkpoint_names.runtime_to_checkpoint(runtime)
   w,counts=constructed(handles[index[key]].get_tensor(key),matrices,scales,stem,config,eigens)
   if not torch.isfinite(w).all():raise ValueError(f'Nonfinite: {stem}')
   yield stem,runtime,key,w,counts

def apply(model,matrices,scales,config):
 start=time.monotonic();eigens=None
 if config['family'] in ['C2','C3']:
  summary=json.loads((OUT/'statistics/S0_summary.json').read_text());assert summary['state']=='complete'
  eigens={key:torch.load(OUT/'statistics/eigen'/f'{key}.pt',map_location='cpu',weights_only=True) for key in summary['groups']}
 params=dict(model.named_parameters());masked={'MFA_multi':0,'MFA_text':0};n=0
 with torch.no_grad():
  for stem,runtime,key,w,counts in iter_weights(matrices,scales,config,eigens):
   assert params[runtime].shape==w.shape
   params[runtime].copy_(w.to(params[runtime].device));n+=1
   for k,v in counts.items():masked[k]+=v
 assert n==274
 return dict(seconds=time.monotonic()-start,layers=n,masked_parameters=masked)

def exact_guard():
 frozen();torch.set_num_threads(4);start=time.monotonic();matrices,scales=load_lora_matrices();cfg=dict(family='C1',c=1)
 dirs={'MFA':REPO/'model/unlearned_model/llava-onevision/split_ratio_5/MFA_unsculpted/sculpted','SMFA':SCULPTED}
 idx={n:json.loads((d/'model.safetensors.index.json').read_text())['weight_map'] for n,d in dirs.items()};counts={'MFA':0,'SMFA':0}
 with ExitStack() as stack:
  handles={n:{s:stack.enter_context(safe_open(d/s,framework='pt',device='cpu')) for s in set(idx[n].values())} for n,d in dirs.items()}
  for stem,runtime,key,w,_ in iter_weights(matrices,scales,cfg):
   source='MFA' if group_key(stem) else 'SMFA';expected=handles[source][idx[source][key]].get_tensor(key)
   if not torch.equal(w,expected):raise ValueError(f'C1 exact guard failed: {key} {source}')
   counts[source]+=1
   if sum(counts.values())%28==0:print('checked',counts,flush=True)
 assert counts=={'MFA':168,'SMFA':106},counts
 report=dict(state='passed',at=now(),seconds=time.monotonic()-start,counts=counts,all_elementwise_equal=True,code_sha256=sha256(Path(__file__)))
 atomic_json(OUT/'c1_exact_guard.json',report);print(report,flush=True)

if __name__=='__main__':exact_guard()
