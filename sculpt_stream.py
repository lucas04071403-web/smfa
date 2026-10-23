import argparse, gc, torch
from peft import PeftModel
from transformers import LlavaOnevisionForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from peft.tuners.lora import Linear as LoraLinear

def main(a):
 cls=Qwen2_5_VLForConditionalGeneration if 'qwen2.5' in a.model_id.lower() else LlavaOnevisionForConditionalGeneration
 m=cls.from_pretrained(a.original_dir,torch_dtype=torch.float16,device_map={'':0},low_cpu_mem_usage=True,local_files_only=True)
 m=PeftModel.from_pretrained(m,a.MFA_multi,adapter_name='MFA_multi'); m.load_adapter(a.MFA_text,adapter_name='MFA_text'); m.load_adapter(a.RA,adapter_name='RA')
 names={n.replace('base_model.model.',''):mod for n,mod in m.named_modules() if isinstance(mod,LoraLinear)}
 deltas={}
 for ad in ('RA','MFA_multi','MFA_text'):
  m.set_adapter(ad); d={}
  for n,mod in m.named_modules():
   if isinstance(mod,LoraLinear):
    x=n.replace('base_model.model.',''); aa=mod.lora_A[ad].weight.detach().cpu(); bb=mod.lora_B[ad].weight.detach().cpu(); d[x]=mod.scaling[ad]*(bb@aa)
  deltas[ad]=d
 masks={}
 for kind,k in (('MFA_multi',a.multi_k),('MFA_text',a.text_k)):
  masks[kind]={}
  for n,x in deltas['RA'].items():
   y=deltas[kind][n]; ratio=torch.linalg.vector_norm(y)/(torch.linalg.vector_norm(x)+1e-12); masks[kind][n]=((x*y)<0)&(k*ratio*torch.abs(x)<torch.abs(y))
 print('masks ready',flush=True)
 base=m.unload();
 with torch.no_grad():
  for n,mod in base.named_modules():
   if n in names:
    w=mod.weight.data
    x=deltas['MFA_multi'][n].to(w.device); y=deltas['MFA_text'][n].to(w.device)
    x[masks['MFA_multi'][n].to(w.device)]=0; y[masks['MFA_text'][n].to(w.device)]=0
    mod.weight.data.copy_(w+(x+y)/2)
    del x,y
 gc.collect(); torch.cuda.empty_cache(); base.save_pretrained(a.save_dir); print('saved',a.save_dir,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();
 for n in ('model_id','original_dir','save_dir','MFA_multi','MFA_text','RA'): p.add_argument('--'+n,required=True)
 p.add_argument('--multi_k',type=float,required=True); p.add_argument('--text_k',type=float,required=True); main(p.parse_args())
