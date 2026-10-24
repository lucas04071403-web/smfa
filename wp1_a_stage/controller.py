"""Resumable A-stage supervisor: generation -> frozen selection -> 975 judgments -> report."""
import os,sys,json,time,subprocess,traceback,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wp1_a_stage.common import OUT,now,atomic_json,frozen
from wp1_a_stage.candidates import SPECTRAL
PYTHON='/root/anaconda3/envs/smllmun/bin/python'
def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False

def spawn(script,args,log,gpu=None):
 env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
 if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
 with (OUT/log).open('a') as f:p=subprocess.Popen([PYTHON,str(OUT/script),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 return dict(pid=p.pid,script=script,args=args,log=log,gpu=gpu,started_at=now())
def complete(names):return all((OUT/'candidates'/name/'metrics.json').exists() for name in names)
def main():
 frozen();lock=(OUT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 jobs=json.loads((OUT/'initial_jobs.json').read_text())['jobs'];spectral={0:['C2_l10','C2_l1000'],1:['C2_l100','C3_hard']};running={};start=now();retries={}
 while True:
  statsready=(OUT/'statistics/S0_summary.json').exists()
  if statsready and not (OUT/'spectral_validation.json').exists():
   with (OUT/'spectral_validation.log').open('a') as stream:subprocess.run([PYTHON,str(OUT/'validate_spectral.py')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
  if statsready:assert json.loads((OUT/'spectral_validation.json').read_text())['state']=='passed'
  if not statsready:
   sj=json.loads((OUT/'statistics_job.json').read_text())
   if not alive(sj['pid']):raise RuntimeError('Statistics job ended without complete S0; inspect statistics log and resume')
  for job in jobs:
   gpu=job['gpu'];names=job['candidates']
   if not complete(names) and not alive(job['pid']):
    retries[gpu]=retries.get(gpu,0)+1
    if retries[gpu]>2:raise RuntimeError(f'GPU{gpu} initial worker failed repeatedly')
    replacement=spawn('evaluate.py',names,f'gpu{gpu}_controller_resume{retries[gpu]}.log',gpu);job['pid']=replacement['pid']
   if complete(names) and not alive(job['pid']) and statsready and not complete(spectral[gpu]):
    old=running.get(gpu)
    if old and not alive(old['pid']):raise RuntimeError(f'Spectral GPU{gpu} failed; resumable cache retained')
    if not old:running[gpu]=spawn('evaluate.py',spectral[gpu],f'gpu{gpu}_spectral.log',gpu)
  statuses={}
  for p in sorted((OUT/'candidates').glob('*/status.json')):statuses[p.parent.name]=json.loads(p.read_text())
  atomic_json(OUT/'status.json',dict(state='running',phase='development_generation',started_at=start,updated_at=now(),initial_jobs=jobs,spectral_jobs=running,statistics_complete=statsready,candidates=statuses))
  if all(complete(j['candidates']) for j in jobs) and complete([x['name'] for x in SPECTRAL]):break
  time.sleep(30)
 from wp1_a_stage.analyze import select,closeout
 while True:
  selection=select()
  if selection['state']!='requires_bracketing':break
  c=selection['next_c'];name='C1_extra'+str(c).replace('.','p');atomic_json(OUT/'extra_candidates'/f'{name}.json',dict(name=name,family='C1',c=c,reason='Frozen C1 bracketing rule; no extrapolation'))
  job=spawn('evaluate.py',[name],name+'.log',0);atomic_json(OUT/'status.json',dict(state='running',phase='C1_bracketing',updated_at=now(),job=job))
  while alive(job['pid']):time.sleep(30)
  if not complete([name]):raise RuntimeError('C1 extension failed')
 if not (OUT/'factuality/scores.json').exists():
  job=spawn('judge.py',[],'factuality.log');atomic_json(OUT/'status.json',dict(state='running',phase='975_concurrent_judgments',updated_at=now(),job=job))
  while alive(job['pid']):time.sleep(30)
  if not (OUT/'factuality/scores.json').exists():raise RuntimeError('Judging failed; use cache and resume')
 result=closeout();state='complete' if selection['state']=='selected' else 'complete_with_unbracketed_lambda'
 atomic_json(OUT/'status.json',dict(state=state,started_at=start,finished_at=now(),selection_state=selection['state'],selected_lambda=selection.get('selected_lambda'),planning_h_pp=result['planning_h_pp'],margin_pp=result['margin_pp'],report='SMFA_WP1_A阶段_实验记录与结果.md'))
if __name__=='__main__':
 try:main()
 except Exception as e:
  atomic_json(OUT/'status.json',dict(state='failed',updated_at=now(),error=str(e),traceback=traceback.format_exc()));raise
