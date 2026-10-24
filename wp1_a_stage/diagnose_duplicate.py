"""Inspect full duplicate-image activations for the pair that tripped pilot tolerance."""
import os,sys,json,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wp1_a_stage.activation_stats import make_inputs
from wp1_a_stage.common import OUT,DATA,rows,atomic_json,now
from wp1_review.memory_construct import ORIGINAL,BASE_MODEL
from transformers import AutoProcessor,LlavaOnevisionForConditionalGeneration
samples=json.loads((OUT/'b1_selection.json').read_text())['selected'];target=94;first=next(i for i,s in enumerate(samples[:target]) if s['image_sha256']==samples[target]['image_sha256']);print('pair',first,target,flush=True)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
model=LlavaOnevisionForConditionalGeneration.from_pretrained(str(ORIGINAL),torch_dtype=torch.float16,device_map={'':'cuda:0'},low_cpu_mem_usage=True,local_files_only=True).eval();processor=AutoProcessor.from_pretrained(str(BASE_MODEL),local_files_only=True)
f=rows(DATA/'unlearning_data/retain_few_shot_5/train-00000-of-00001.parquet');byid={r.ID:r for _,r in f.iterrows()};captured=[];context={};current={}
handles=[]
for layer in [0,26,27]:
 for group,module in [('attn',model.model.language_model.layers[layer].self_attn.q_proj),('o',model.model.language_model.layers[layer].self_attn.o_proj),('mlp',model.model.language_model.layers[layer].mlp.gate_proj)]:
  def hook(module,args,key=f'layer_{layer}_{group}'):
   current[key]=args[0][0][context['mask']].detach().cpu().float()
  handles.append(module.register_forward_pre_hook(hook))
inputs=[]
for i in [first,target]:
 inp=make_inputs(processor,samples[i],byid).to('cuda');context['mask']=inp.input_ids[0]==model.config.image_token_id;inputs.append(inp.input_ids[0][:int(context['mask'].nonzero()[-1])+1].cpu())
 with torch.inference_mode():model.model(**inp,use_cache=False)
 captured.append(current.copy());current.clear()
assert torch.equal(*inputs)
result={}
for key,a in captured[0].items():
 b=captured[1][key];diff=a-b;result[key]=dict(shape=list(a.shape),max_abs=float(diff.abs().max()),reference_max_abs=float(a.abs().max()),relative_l2=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(a)),rms=float(diff.square().mean().sqrt()),exact_fraction=float((a==b).float().mean()))
atomic_json(OUT/'duplicate_activation_diagnostic.json',dict(at=now(),pair_indices=[first,target],image_sha256=samples[first]['image_sha256'],tokens_through_last_image_equal=True,groups=result));print(json.dumps(result,indent=2))
