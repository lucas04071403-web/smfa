"""Synthetic nonlinear matching/budget checks; no model generation or real caches."""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
from wp1_supplement import controller as c,analyze as a
from wp1_supplement.common import OUT,now,atomic_json,sha256

def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    actual={};calls=[];force_text_failure=False
    def summary(name):
        if name=='SMFA':return dict(forget_text=.517)
        if name=='Original':return dict(forget_text=.744)
        if name=='Base':return dict(forget_text=.373)
        cfg=actual[name];strength=cfg['alpha'];x=.7-.28*strength**2
        return dict(name=name,forget_combined=x,retain_image=.72-.07*strength,
                    forget_text=.60 if force_text_failure else .517+.3*(x-protocol['target']))
    def evaluate(configs,phase):
        calls.extend(configs)
        for cfg in configs:actual[cfg['name']]=cfg
    base=dict(family='C2',lambda_rel=10)
    initial=[dict(base,name='toy_half',alpha=.5),dict(base,name='toy_one',alpha=1.)]
    actual.update({cfg['name']:cfg for cfg in initial})
    checks={}
    with tempfile.TemporaryDirectory(prefix='wp1-workflow-check-') as directory:
        dest=Path(directory)
        with patch.object(c,'OUT',dest),patch.object(c,'frozen',lambda:(protocol,{})),patch.object(c,'summary',summary),patch.object(a,'summary',summary),patch.object(c,'wait_for',evaluate):
            refined=c.refine_bracket(initial,'C2',base,'synthetic')
            assert len(refined)==4 and len(calls)==2
            checks['midpoint_budget_and_nonlinear_bracketing']=True
            match=c.match(refined,'C2',base)
            assert match['state']=='matched'
            assert protocol['hit_range'][0]<=match['summary']['forget_combined']<=protocol['hit_range'][1]
            checks['nonlinear_actual_validation_hits_interval']=True
            n=len(calls);again=c.match(refined,'C2',base)
            assert again==match and len(calls)==n
            checks['resume_does_not_repeat_validated_match']=True
            bad=[dict(family='C3',name='unbracketed',alpha=.1)]
            actual[bad[0]['name']]=bad[0]
            rejected=c.match(bad,'C3',dict(family='C3'))
            assert rejected['state']=='unmatchable' and not rejected['attempts']
            checks['no_extrapolation_or_forced_winner']=True
            soft=dict(state='matched',selected_config=dict(name='soft'),summary=dict(retain_image=.50))
            hard=dict(state='matched',selected_config=dict(name='hard'),summary=dict(retain_image=.509))
            assert c.choose(soft,hard,10)['selected_family']=='C2'
            hard['summary']['retain_image']=.511
            assert c.choose(soft,hard,10)['selected_family']=='C3'
            checks['soft_tie_rule']=True
            (dest/'matching/C2.json').unlink()
            (dest/'matching/C2_progress.json').unlink()
            for path in (dest/'plans').glob('C2_target*.json'):path.unlink()
            force_text_failure=True
            result=c.match(refined,'C2',base)
            assert result['state']=='unmatchable' and len(result['attempts'])==3
            plans=[json.loads(p.read_text()) for p in sorted((dest/'plans').glob('C2_target*.json'))]
            assert plans[0]['aim_target']==protocol['target'] and all(p['aim_target']==.44129 for p in plans[1:])
            checks['text_monitor_failure_bounded_and_stronger_aim']=True
    report=dict(state='passed',at=now(),checks=checks,no_GPU_or_real_candidate_outputs=True,
                code_sha256={p.name:sha256(p) for p in (OUT/'workflow_checks.py',OUT/'controller.py',OUT/'analyze.py')})
    atomic_json(OUT/'workflow_unit_checks.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
