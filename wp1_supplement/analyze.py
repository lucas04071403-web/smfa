"""Enriched C1 selection and paired person bootstrap on development caches."""
import json
from collections import defaultdict
import numpy as np
from rouge_score.rouge_scorer import RougeScorer
from wp1_supplement.common import OUT,PARENT,now,atomic_json,sha256,frozen,result_dir,summary

OLD_C1=['C1_c0p25','C1_c0p5','C1_c0p75','C1_c1p0']
NEW_C1=['C1_c0p8','C1_c0p85','C1_c0p9','C1_c0p95']
C2=['C2_l10','C2_l100','C2_l1000']

def c1_config(name):
    if not name.startswith('C1_c'):
        manifest=result_dir(name)/'manifest.json'
        cfg=json.loads(manifest.read_text())['config'];assert cfg['family']=='C1'
        return cfg
    return dict(name=name,family='C1',c=float(name.removeprefix('C1_c').replace('p','.')))

def points(configs):
    unique={}
    for cfg in configs:
        s=summary(cfg['name']);p=dict(s,parameter=cfg.get('c',cfg.get('alpha',cfg.get('k_multi'))))
        x=s['forget_combined']
        if x not in unique or (p['retain_image'],-p['parameter'])>(unique[x]['retain_image'],-unique[x]['parameter']):unique[x]=p
    return sorted(unique.values(),key=lambda p:p['forget_combined'])

def bracket(configs,target):
    curve=points(configs)
    below=[p for p in curve if p['forget_combined']<=target]
    above=[p for p in curve if p['forget_combined']>=target]
    return (below[-1],above[0]) if below and above else None

def interpolate(configs,target,field='retain_image'):
    b=bracket(configs,target)
    if b is None:return None
    lo,hi=b
    if lo['forget_combined']==hi['forget_combined']:return lo[field]
    w=(target-lo['forget_combined'])/(hi['forget_combined']-lo['forget_combined'])
    return lo[field]+w*(hi[field]-lo[field])

def selection(names,filename):
    configs=[c1_config(n) for n in names]
    old=json.loads((PARENT/'selection.json').read_text())
    comparisons=[]
    for name,lam in zip(C2,(10,100,1000)):
        s=summary(name);baseline=interpolate(configs,s['forget_combined'])
        comparisons.append(dict(name=name,lambda_rel=lam,forget_combined_rouge=s['forget_combined'],retain_image_rouge=s['retain_image'],
                                c1_interpolated_rouge=baseline,uplift=None if baseline is None else s['retain_image']-baseline))
    assert all(c['uplift'] is not None for c in comparisons),'Unbracketed lambda comparison; no forced selection'
    best=max(c['uplift'] for c in comparisons)
    selected=min((c for c in comparisons if best-c['uplift']<=1e-8),key=lambda c:c['lambda_rel'])
    others={}
    for name in ('SMFA','C0_k1','C0_k10'):
        s=summary(name);baseline=interpolate(configs,s['forget_combined'])
        others[name]=dict(c1_interpolated_rouge=baseline,uplift=None if baseline is None else s['retain_image']-baseline)
    result=dict(state='selected',at=now(),curve_names=names,c1_curve=points(configs),comparisons=comparisons,
                selected_lambda=selected['lambda_rel'],selected_candidate=selected['name'],comparisons_vs_c1=others,
                original_selection=dict(source_sha256=sha256(PARENT/'selection.json'),selected_lambda=old['selected_lambda'],comparisons=old['comparisons']),
                rule='Original 6.1. Max retain-image uplift over observed C1 interpolation at each C2 forget-combined x. No extrapolation; lambda ties 1e-8 choose smaller.')
    atomic_json(OUT/filename,result)
    return result

def person_values(name,split):
    path=result_dir(name)/f'{split}_generation_results.json'
    cache=OUT/'person_metrics'/f'{name}_{split}.json'
    digest=sha256(path)
    if cache.exists():
        saved=json.loads(cache.read_text())
        if saved['source_sha256']==digest:return saved['ids'],np.array(saved['values']),digest
    scorer=RougeScorer(['rougeL'],use_stemmer=True);groups=defaultdict(list)
    for q in json.loads(path.read_text())['Generation_Questions']:
        kind=q['question type']
        if (kind.endswith('memory') if split=='forget' else kind=='image_memory'):
            groups[q['person_id']].append(scorer.score(q['ground_truth'],q['generated_answer'])['rougeL'].fmeasure)
    ids=sorted(groups);n=13 if split=='forget' else 7
    assert len(ids)==(25 if split=='forget' else 100) and all(len(groups[pid])==n for pid in ids)
    values=np.array([np.mean(groups[pid]) for pid in ids])
    _,partition=frozen();assert ids==sorted(partition[split]['F1_development' if split=='forget' else 'development'])
    atomic_json(cache,dict(ids=ids,values=values.tolist(),source_sha256=digest))
    return ids,values,digest

def interpolation_draws(xs,ys,target):
    result=np.full(len(target),np.nan)
    for i in range(len(target)):
        unique={}
        for x,y in zip(xs[i],ys[i]):unique[x]=max(unique.get(x,-np.inf),y)
        curve=sorted(unique.items())
        if curve[0][0]<=target[i]<=curve[-1][0]:result[i]=np.interp(target[i],*np.asarray(curve).T)
    return result

def bootstrap(names,extra_comparisons=(),filename='c1_bootstrap.json'):
    protocol,_=frozen();replicates=protocol['bootstrap']['replicates'];seed=protocol['bootstrap']['seed']
    comparisons=['SMFA','C0_k1','C0_k10']+C2+list(extra_comparisons)
    allnames=list(dict.fromkeys(names+comparisons))
    rng=np.random.default_rng(seed)
    fdraw=rng.integers(0,25,(replicates,25));rdraw=rng.integers(0,100,(replicates,100))
    xs,ys,sources={},{},{}
    for name in allnames:
        for split,draw,store in [('forget',fdraw,xs),('retain',rdraw,ys)]:
            ids,values,digest=person_values(name,split);store[name]=values[draw].mean(1)
            sources[str((result_dir(name)/f'{split}_generation_results.json').relative_to(OUT.parent))]=digest
            label='forget_combined' if split=='forget' else 'retain_image'
            assert abs(values.mean()-summary(name)[label])<1e-10
    xcurve=np.column_stack([xs[n] for n in names]);ycurve=np.column_stack([ys[n] for n in names]);reports={};lifts=[]
    configs=[c1_config(n) for n in names]
    for name in comparisons:
        values=ys[name]-interpolation_draws(xcurve,ycurve,xs[name]);finite=values[np.isfinite(values)]
        s=summary(name);baseline=interpolate(configs,s['forget_combined'])
        reports[name]=dict(estimate=None if baseline is None else s['retain_image']-baseline,
                           valid_replicates=len(finite),unbracketed_replicates=replicates-len(finite),
                           conditional_ci95=np.quantile(finite,[.025,.975]).tolist() if len(finite) else None,
                           conditional_ci90=np.quantile(finite,[.05,.95]).tolist() if len(finite) else None,
                           conditional_fraction_above_0_01=float((finite>.01).mean()) if len(finite) else None)
        if name in C2:lifts.append(values)
    matrix=np.column_stack(lifts);valid=np.isfinite(matrix).all(1);winners=np.argmax(matrix[valid],axis=1)
    result=dict(state='complete',at=now(),seed=seed,replicates=replicates,curve_names=names,comparisons=reports,
                lambda_ranking=dict(fully_bracketed_replicates=int(valid.sum()),conditional_selection_frequencies={n:float((winners==i).mean()) for i,n in enumerate(C2)}),
                method='Person bootstrap; independent forget/retain samples, paired across models within each cohort; reinterpolate each draw; no extrapolation.',
                limitation=protocol['bootstrap']['limitation'],interval_note='Intervals exclude curvature error, candidate-selection error and factuality scoring uncertainty. Frequencies are not probabilities of population optimality.',
                source_sha256=sources)
    atomic_json(OUT/filename,result);return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--bootstrap',action='store_true');args=p.parse_args()
    names=json.loads((OUT/'c1_selection.json').read_text())['curve_names']
    if args.bootstrap:bootstrap(names)
    else:selection(names,'c1_selection.json')
