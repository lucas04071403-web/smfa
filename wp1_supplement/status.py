"""Read-only concise status; safe to call while workers are running."""
import json
from pathlib import Path
from wp1_supplement.common import OUT

def main():
    def read(path):
        try:return json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):return None
    result={'workflow':read(OUT/'workflow_status.json'),'controller_failure':read(OUT/'controller_failure.json')}
    result['candidates']={}
    for path in sorted((OUT/'candidates').glob('*/status.json')):
        s=read(path)
        if s:result['candidates'][path.parent.name]={k:s[k] for k in ('state','gpu','completed_questions','total_questions','elapsed_generation_seconds','generation_seconds','updated_at') if k in s}
    result['appearance']={}
    for path in sorted((OUT/'appearance').glob('*/*/status.json')):
        s=read(path)
        if s:result['appearance']['/'.join(path.parts[-3:-1])]={k:s[k] for k in ('state','completed','total','generation_seconds','updated_at') if k in s}
    result['appearance_summary']=read(OUT/'appearance/summary.json')
    result['diagnostics']={name:read(OUT/'diagnostics'/name/'progress.json') for name in ('pilot','D_fm','D_rh','D_fi')}
    result['diagnostic_status']=read(OUT/'diagnostics/status.json')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
