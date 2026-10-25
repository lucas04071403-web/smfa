"""Resumable bounded C1 enrichment -> lambda freeze -> B1 matching workflow."""
import os,sys,json,time,subprocess,traceback,fcntl,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wp1_supplement.common import OUT,PARENT,now,atomic_json,sha256,frozen,queue,result_dir,summary
from wp1_supplement.analyze import OLD_C1,NEW_C1,c1_config,points,bracket,interpolate,selection,bootstrap

STATUS_LOCK=threading.Lock()
PHASES={}

def write_progress(phase,configs,details):
    with STATUS_LOCK:
        PHASES[phase]=dict(complete=sum(complete(c) for c in configs),total=len(configs),progress=details)
        atomic_json(OUT/'workflow_status.json',dict(state='running',phase=phase,at=now(),**PHASES[phase],parallel_phases=PHASES))

def complete(cfg):return (result_dir(cfg['name'])/'metrics.json').exists()

def wait_for(configs,phase):
    for i,cfg in enumerate(configs):
        if not complete(cfg):queue(cfg,10 if phase.startswith('C1') else 100)
    while not all(complete(cfg) for cfg in configs):
        progress={}
        for cfg in configs:
            status=OUT/'candidates'/cfg['name']/'status.json'
            if status.exists():
                s=json.loads(status.read_text())
                assert s.get('state')!='failed',f"Candidate failed {cfg['name']}: {s.get('error')}"
                progress[cfg['name']]=dict(state=s['state'],completed=s.get('completed_questions',0))
        write_progress(phase,configs,progress)
        time.sleep(10)

def refine_bracket(configs,family,base,phase):
    protocol,_=frozen();target=protocol['target'];work=list(configs);decisions=[]
    for index in range(2):
        b=bracket(work,target)
        if b is None:break
        lo,hi=b;gap=hi['forget_combined']-lo['forget_combined']
        if gap<=.1:break
        parameter=(lo['parameter']+hi['parameter'])/2
        key='c' if family=='C1' else 'k_multi' if family=='C0' else 'alpha'
        cfg=dict(base,name=f'{family}_bracket{index+1}',**{key:parameter})
        if family=='C0':cfg['k_text']=2*parameter
        record=dict(at=now(),index=index+1,target=target,bracket=[lo,hi],gap=gap,config=cfg)
        path=OUT/'plans'/f'{family}_bracket{index+1}.json'
        if path.exists():
            old=json.loads(path.read_text());assert old['config']==cfg;record=old
        else:atomic_json(path,record)
        decisions.append(record);wait_for([cfg],phase);work.append(cfg)
    atomic_json(OUT/'plans'/f'{family}_bracketing.json',dict(at=now(),configs=work,decisions=decisions,final_bracket=bracket(work,target),max_additions=2))
    return work

def text_delta(s):
    smfa,orig,base=summary('SMFA'),summary('Original'),summary('Base')
    return 100*(s['forget_text']-smfa['forget_text'])/(orig['forget_text']-base['forget_text'])

def eligible(s):
    protocol,_=frozen();low,high=protocol['hit_range']
    return bool(low<=s['forget_combined']<=high and text_delta(s)<=protocol['text_monitor_pp'])

def match(configs,family,base):
    output=OUT/'matching'/f'{family}.json'
    if output.exists():return json.loads(output.read_text())
    protocol,_=frozen();target=protocol['target'];work=list(configs);attempts=[]
    if bracket(work,target) is None:
        result=dict(state='unmatchable',at=now(),family=family,reason='Target outside observed strength range; no extrapolation.',configs=work,attempts=[])
        atomic_json(output,result);return result
    for index in range(3):
        name=f'{family}_target{index}'
        plan=OUT/'plans'/f'{name}.json'
        key='c' if family=='C1' else 'k_multi' if family=='C0' else 'alpha'
        if plan.exists():
            proposal=json.loads(plan.read_text());cfg=proposal['config'];target=proposal['aim_target']
        else:
            parameter=interpolate(work,target,field='parameter')
            if parameter is None:break
            cfg=dict(base,name=name,**{key:parameter})
            if family=='C0':cfg['k_text']=2*parameter
            proposal=dict(at=now(),config=cfg,aim_target=target,source_configs=work,source_bracket=bracket(work,target),
                          predicted_text_monitor_pp=100*(interpolate(work,target,'forget_text')-summary('SMFA')['forget_text'])/(summary('Original')['forget_text']-summary('Base')['forget_text']))
            atomic_json(plan,proposal)
        wait_for([cfg],f'B1 actual matching {family}')
        s=summary(name);passed=eligible(s)
        attempt=dict(config=cfg,summary=s,text_monitor_difference_pp=text_delta(s),eligible=passed,aim_target=target,
                     in_hit_range=protocol['hit_range'][0]<=s['forget_combined']<=protocol['hit_range'][1])
        attempts.append(attempt);work.append(cfg)
        atomic_json(OUT/'matching'/f'{family}_progress.json',dict(at=now(),attempts=attempts))
        if passed:
            result=dict(state='matched',at=now(),family=family,selected_config=cfg,summary=s,text_monitor_difference_pp=text_delta(s),
                        attempts=attempts,configs=work,interpolated_retain_at_exact_target=interpolate(work,protocol['target']))
            atomic_json(output,result);return result
        if text_delta(s)>protocol['text_monitor_pp']:target=.44129
    result=dict(state='unmatchable',at=now(),family=family,reason='Actual validation failed hit interval or text monitor within three actual attempts.',
                attempts=attempts,configs=work)
    atomic_json(output,result);return result

def choose(soft,hard,lambda_rel):
    protocol,_=frozen();softok=soft['state']=='matched';hardok=hard['state']=='matched';chosen=None
    difference=None;actual_difference=None
    if softok:assert soft['interpolated_retain_at_exact_target'] is not None
    if hardok:assert hard['interpolated_retain_at_exact_target'] is not None
    if softok and hardok:
        actual_difference=soft['summary']['retain_image']-hard['summary']['retain_image']
        difference=soft['interpolated_retain_at_exact_target']-hard['interpolated_retain_at_exact_target']
        chosen='C2' if abs(difference)<protocol['soft_tie_rouge'] or difference>0 else 'C3'
    elif softok:chosen='C2'
    elif hardok:chosen='C3'
    result=dict(state='selected' if chosen else 'no_eligible_operator',at=now(),selected_family=chosen,
                selected_lambda=lambda_rel if chosen=='C2' else None,
                selected_config=(soft if chosen=='C2' else hard)['selected_config'] if chosen else None,
                soft_minus_hard_retain_image=difference,soft_tie_threshold=protocol['soft_tie_rouge'],soft=soft,hard=hard,
                comparison_basis='v3 section 6.2: interpolate both actually validated curves at the same fixed target',
                actual_soft_minus_hard_retain_image=actual_difference,
                interpretation='Development ROUGE selection only; not a test-set or factuality superiority claim.')
    atomic_json(OUT/'B1_operator_selection.json',result);return result

def launch_bootstrap(names):
    code="from wp1_supplement.analyze import bootstrap; import json; from wp1_supplement.common import OUT; bootstrap(json.loads((OUT/'c1_selection.json').read_text())['curve_names'])"
    if (OUT/'c1_bootstrap.json').exists():return None
    with (OUT/'logs/bootstrap.log').open('ab',buffering=0) as log:
        return subprocess.Popen([sys.executable,'-u','-c',code],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)

def main():
    frozen();OUT.joinpath('plans').mkdir(exist_ok=True)
    correction=OUT/'operator_comparison_correction.json'
    assert (OUT/'OPERATOR_COMPARISON.sha256').read_text()==sha256(correction)+'  operator_comparison_correction.json\n'
    lock=(OUT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    initial=[c1_config(n) for n in OLD_C1+NEW_C1]
    wait_for(initial,'C1 four new points')
    if not (OUT/'c1_eight_point_selection.json').exists():selection(OLD_C1+NEW_C1,'c1_eight_point_selection.json')
    c1=refine_bracket(initial,'C1',dict(family='C1'),'C1 target bracket refinement')
    if (OUT/'c1_selection.json').exists():
        selected=json.loads((OUT/'c1_selection.json').read_text());assert selected['curve_names']==[c['name'] for c in c1]
    else:selected=selection([c['name'] for c in c1],'c1_selection.json')
    lam=selected['selected_lambda']
    b1_inputs=OUT/'B1_selection_inputs.json'
    inputs=dict(frozen_at=now(),selected_lambda=lam,c1_selection_sha256=sha256(OUT/'c1_selection.json'),
                c1_names=[c['name'] for c in c1],supplement_freeze_sha256=sha256(OUT/'FREEZE_supplement.sha256'),
                note='Lambda locked before any new B1 alpha generation; diagnostics cannot change this selection.')
    if b1_inputs.exists():
        saved=json.loads(b1_inputs.read_text());assert saved['selected_lambda']==lam and saved['c1_names']==inputs['c1_names']
    else:
        atomic_json(b1_inputs,inputs)
        (OUT/'B1_selection_inputs.sha256').write_text(sha256(b1_inputs)+'  B1_selection_inputs.json\n')
    boot=launch_bootstrap([c['name'] for c in c1])
    softbase=dict(family='C2',lambda_rel=lam);hardbase=dict(family='C3')
    soft=[dict(softbase,name=f'C2_l{lam}',alpha=1)]+[dict(softbase,name='C2_a'+str(a).replace('.','p'),alpha=a) for a in (1.25,1.5,2.,3.)]
    hard=[dict(hardbase,name='C3_hard',alpha=1)]+[dict(hardbase,name='C3_a'+str(a).replace('.','p'),alpha=a) for a in (1.5,2.,3.)]
    # Enqueue all B1 grid jobs first so two workers remain occupied during matches.
    for cfg in soft+hard:
        if not complete(cfg):queue(cfg,100)
    c0=[dict(name='C0_k1',family='C0',k_multi=1,k_text=2),dict(name='SMFA',family='C0',k_multi=5,k_text=10),dict(name='C0_k10',family='C0',k_multi=10,k_text=20)]
    def finish_soft():
        refined=refine_bracket(soft,'C2',softbase,'B1 soft target bracket refinement')
        return match(refined,'C2',softbase)
    def finish_hard():
        protocol,_=frozen();work=list(hard)
        if bracket(work,protocol['target']) is None and min(summary(c['name'])['forget_combined'] for c in work)>protocol['target']:
            extensions=[dict(hardbase,name=f'C3_a{a}',alpha=a) for a in (5,8)]
            wait_for(extensions,'B1 hard projection extensions 5 and 8');work+=extensions
        refined=refine_bracket(work,'C3',hardbase,'B1 hard target bracket refinement')
        return match(refined,'C3',hardbase)
    # Independent families can refine/match concurrently, keeping both GPU queues busy.
    with ThreadPoolExecutor(max_workers=4) as pool:
        c0future=pool.submit(match,c0,'C0',dict(family='C0'))
        c1future=pool.submit(match,c1,'C1',dict(family='C1'))
        wait_for(soft+hard,'B1 soft and hard strength grids')
        softfuture=pool.submit(finish_soft);hardfuture=pool.submit(finish_hard)
        c0match,c1match=c0future.result(),c1future.result()
        softmatch,hardmatch=softfuture.result(),hardfuture.result()
    result=choose(softmatch,hardmatch,lam)
    atomic_json(OUT/'B1_baselines.json',dict(at=now(),C0=c0match,C1=c1match))
    if boot is not None:assert boot.wait()==0,'Bootstrap process failed'
    while True:
        dpath=OUT/'diagnostics/status.json';apath=OUT/'appearance/summary.json'
        diag_complete=dpath.exists() and json.loads(dpath.read_text()).get('state')=='complete'
        if diag_complete and apath.exists():break
        atomic_json(OUT/'workflow_status.json',dict(state='running',phase='B1 complete; waiting for diagnostics and appearance',at=now()))
        time.sleep(10)
    atomic_json(OUT/'workflow_status.json',dict(state='running',phase='completion audit and report',at=now()))
    subprocess.run([sys.executable,'-u','-m','wp1_supplement.finalize'],cwd=str(ROOT),check=True)
    atomic_json(OUT/'workflow_status.json',dict(state='complete',phase='B1 complete',finished_at=now(),selected_family=result['selected_family'],selected_lambda=lam))
    (OUT/'STOP_WORKERS').write_text(now()+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        atomic_json(OUT/'controller_failure.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc()))
        raise
