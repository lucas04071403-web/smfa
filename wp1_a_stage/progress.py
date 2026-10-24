"""Compact read-only progress for the long-running A-stage jobs."""
import json,sys
from pathlib import Path
from datetime import datetime,timezone
p=Path(__file__).resolve().parent
s=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {}
result={'at':datetime.now(timezone.utc).isoformat(),'controller':s.get('state'),'phase':s.get('phase'),'candidates':{}}
for f in sorted((p/'candidates').glob('*/status.json')):
 d=json.loads(f.read_text());result['candidates'][f.parent.name]={'state':d['state'],'done':d.get('completed_questions',0),'total':2375,'seconds':round(d.get('elapsed_generation_seconds',d.get('generation_seconds',0)),1)}
f=p/'statistics/progress.json'
if f.exists():
 d=json.loads(f.read_text());started=json.loads((p/'statistics_job.json').read_text())['started_at'];fresh=d['at']>=started
 result['statistics']={'done':d['completed'] if fresh else 0,'total':d['total'],'seconds':round(d['elapsed_seconds'],1) if fresh else 0,'state':'collecting_or_eigen' if fresh else 'pilot_recheck_before_restart','eigens_complete':(p/'statistics/S0_summary.json').exists()}
f=p/'factuality/status.json'
if f.exists():result['factuality']=json.loads(f.read_text())
if s.get('state')=='failed':result['error']=s.get('error')
print(json.dumps(result,ensure_ascii=False))
