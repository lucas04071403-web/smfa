"""Check the actual 84 eigensystems and LoRA projections before C2/C3 generation."""
import sys,json,time,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wp1_a_stage.common import OUT,atomic_json,sha256,now,frozen
from wp1_a_stage.candidates import group_key,correct_a
from wp1_review.memory_construct import load_lora_matrices,layer_stems

def main():
 frozen();torch.set_num_threads(4);start=time.monotonic();summary=json.loads((OUT/'statistics/S0_summary.json').read_text());assert summary['state']=='complete' and len(summary['groups'])==84 and summary['samples']==128
 records=summary['sample_records'];assert len(records)==128
 expected=dict(n_prefix=128,n_v_repeated=sum(x['visual_tokens'] for x in records),n_v_unique=sum(x['visual_tokens'] for x in records if x['first_image']),n_t=sum(x['text_tokens'] for x in records))
 factors,_=load_lora_matrices();stems=layer_stems(factors);report={};max_projected=0.;max_orthogonal=0.
 for key,meta in summary['groups'].items():
  assert meta['counts']==expected
  path=OUT/'statistics/eigen'/f'{key}.pt';assert sha256(path)==meta['eigen_sha256']
  moments=OUT/'statistics/moments'/f'{key}.pt';assert sha256(moments)==meta['moments_sha256']
  spec=torch.load(path,map_location='cpu',weights_only=True);v,e,all_e=spec['vectors'],spec['values'],spec['all_eigenvalues'];r=meta['r']
  assert v.shape==(3584,r) and e.shape==(r,) and 0<r<3584 and torch.isfinite(v).all() and torch.isfinite(all_e).all()
  assert bool((all_e[:-1]>=all_e[1:]).all())
  cumulative=all_e.clamp_min(0).cumsum(0)/all_e.clamp_min(0).sum()
  assert float(cumulative[r-1])>=.9-1e-7 and (r==1 or float(cumulative[r-2])<.9+1e-7)
  trace=meta['traces']['V_repeated']+meta['traces']['T']+meta['counts']['n_prefix']*meta['traces']['P']
  trace_relative_error=abs(float(all_e.sum())-trace)/trace
  assert trace_relative_error<3584*torch.finfo(torch.float32).eps
  matrix=torch.load(moments,map_location='cpu',weights_only=True)
  pmat=matrix['prefix'].T@matrix['prefix'];s0=matrix['V_repeated']+matrix['T']+matrix['n_prefix']*pmat
  indices=torch.linspace(0,r-1,min(8,r)).long();probe_v=v[:,indices];probe_e=e[indices]
  eigen_relative_residual=float(torch.linalg.vector_norm(s0@probe_v-probe_v*probe_e)/torch.linalg.vector_norm(s0@probe_v).clamp_min(1e-12))
  assert eigen_relative_residual<5e-4
  del matrix,pmat,s0
  ortho=float((v.T@v-torch.eye(r)).abs().max());assert ortho<2e-5
  checked=0
  for stem in stems:
   if group_key(stem)!=key:continue
   for name in ['MFA_multi','MFA_text']:
    a=factors[name][stem+'.lora_A.weight'];hard=correct_a(a,v,e);rel=float(torch.linalg.vector_norm(hard@v)/torch.linalg.vector_norm(a).clamp_min(1e-12));change=a-hard;orth=float(torch.linalg.vector_norm(change-(change@v)@v.T)/torch.linalg.vector_norm(a).clamp_min(1e-12))
    assert rel<2e-5 and orth<2e-5
    energies=[float(torch.linalg.vector_norm(correct_a(a,v,e,lam)@v)) for lam in [0,10,100,1000]]
    assert all(x+1e-6>=y for x,y in zip(energies,energies[1:]))
    max_projected=max(max_projected,rel);max_orthogonal=max(max_orthogonal,orth);checked+=1
  report[key]=dict(r=r,orthonormal_max_abs=ortho,lora_factors_checked=checked,trace_relative_error=trace_relative_error,eigen_relative_residual=eigen_relative_residual)
 result=dict(state='passed',at=now(),seconds=time.monotonic()-start,groups=report,counts=expected,max_hard_projection_relative_residual=max_projected,max_orthogonal_complement_relative_change=max_orthogonal,S0_summary_sha256=sha256(OUT/'statistics/S0_summary.json'),code_sha256=sha256(Path(__file__)))
 atomic_json(OUT/'spectral_validation.json',result);print('84 groups and 336 actual LoRA A factors passed',flush=True)
if __name__=='__main__':main()
