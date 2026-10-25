"""Give GPU 1 to frozen baseline checks/diagnostics, then candidate queue."""
import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wp1_supplement.common import OUT,now,atomic_json,frozen

PYTHON=sys.executable

def alive(pid):
    path=Path(f'/proc/{pid}/stat')
    if not path.exists():return False
    return path.read_text().split()[2]!='Z'

def launch(module,logname,record):
    with (OUT/'logs'/logname).open('ab',buffering=0) as log:
        child=subprocess.Popen([PYTHON,'-u','-m',module],cwd=str(ROOT),env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',TOKENIZERS_PARALLELISM='false'),
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(OUT/record,dict(pid=child.pid,started_at=now(),gpu=1,module=module,log='logs/'+logname))
    return child

def main():
    frozen()
    appearance=OUT/'appearance/summary.json'
    if not appearance.exists():
        launcher=OUT/'appearance_launcher.json'
        if launcher.exists() and alive(json.loads(launcher.read_text())['pid']):
            pid=json.loads(launcher.read_text())['pid']
            while alive(pid) and not appearance.exists():time.sleep(5)
            # Summary is written at the very end; wait until CUDA memory is released.
            while alive(pid):time.sleep(2)
            assert appearance.exists(),'Appearance worker exited without completing; see log'
        else:
            child=launch('wp1_supplement.appearance','appearance_gpu1.log','appearance_launcher.json')
            assert child.wait()==0,'Appearance worker failed'
    diagnostic=OUT/'diagnostics/status.json'
    complete=diagnostic.exists() and json.loads(diagnostic.read_text()).get('state')=='complete'
    if not complete:
        child=launch('wp1_supplement.diagnostics','diagnostics_gpu1.log','diagnostics_launcher.json')
        assert child.wait()==0,'Diagnostic worker failed'
    if not (OUT/'STOP_WORKERS').exists():
        child=launch('wp1_supplement.evaluate','gpu1.log','worker_gpu1.json')
        atomic_json(OUT/'gpu1_pipeline_status.json',dict(state='candidate_worker',at=now(),pid=child.pid))
        assert child.wait()==0,'Candidate worker 1 failed'
    atomic_json(OUT/'gpu1_pipeline_status.json',dict(state='complete',at=now()))

if __name__=='__main__':main()
