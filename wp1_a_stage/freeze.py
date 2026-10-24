"""Input-only protocol preparation. Never reads candidate outputs."""
import sys,json,random,math,re,hashlib
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from wp1_review.prepare_wp1 import load_partition,rows,DATA,sha256,atomic_json,normalize
OUT=ROOT/'wp1_a_stage'; SEED=20261023

def main():
 if (OUT/'FREEZE.sha256').exists(): raise RuntimeError('Already frozen; do not overwrite')
 part=load_partition(); frame=rows(DATA/'unlearning_data/retain_few_shot_5/train-00000-of-00001.parquet')
 strata=defaultdict(list)
 for _,row in frame.iterrows():
  for j,q in enumerate(json.loads(row.metadata)):
   strata[q['Type']].append(dict(person_id=row.ID,qa_index=j,type=q['Type'],question=q['Question'],answer=q['Answer'],image_sha256=None if q['Type']=='pure_text' else hashlib.sha256(row['character_image' if q['Type']=='multimodal' else 'ophthalmic_image']['bytes']).hexdigest()))
 counts={k:len(v) for k,v in strata.items()}; quota={k:math.floor(128*n/sum(counts.values())) for k,n in counts.items()}
 for k in sorted(counts,key=lambda k:(-(128*counts[k]/sum(counts.values())-quota[k]),k))[:128-sum(quota.values())]:quota[k]+=1
 rng=random.Random(SEED); selected=[]
 for k in sorted(strata):selected+=rng.sample(sorted(strata[k],key=lambda q:(q['person_id'],q['qa_index'])),quota[k])
 rng.shuffle(selected)
 atomic_json(OUT/'b1_selection.json',dict(seed=SEED,method='proportional strata; largest remainder; sorted type/person/id then Python Random.sample, shared shuffle',source_counts=counts,selected_counts=quota,source_sha256=sha256(DATA/'unlearning_data/retain_few_shot_5/train-00000-of-00001.parquet'),selected=selected))
 templates={
 'B2':[
 ('face_horizontal_position','Relative to the frame width, is the centre of the face left of centre, centred, or right of centre?'),
 ('face_vertical_position','Relative to the frame height, is the centre of the face in the upper, middle, or lower third?'),
 ('face_area_fraction','Does the face itself occupy less than one quarter, about one quarter to one half, or more than half of the picture area?'),
 ('chin_crop','Is the bottom of the chin cut off by the lower edge of this picture?'),
 ('forehead_crop','Is any part of the forehead cut off by the upper edge of this picture?'),
 ('cheek_crop','Is either cheek cut off by a side edge of the picture?'),
 ('head_margin','Is there an empty gap between the top of the head and the upper picture edge?'),
 ('head_roll','Is the line joining the two eyes approximately level, tilted clockwise, or tilted counterclockwise in the picture?')],
 'appearance_holdout':[
 ('colour_mode','Is this picture rendered in colour or entirely in shades of grey?'),
 ('rendering_medium','Does this image look like a photograph or a drawn illustration?'),
 ('text_overlay','Is there any printed text or a watermark overlaid on this picture?'),
 ('frame_border','Is there a visible decorative border enclosing this picture?'),
 ('block_artifacts','Are conspicuous square compression blocks or coarse pixelation visible across the face?'),
 ('image_noise','Does the facial region have noticeable speckled image noise or film grain?')],
 }
 templates={k:[dict(attribute=a,question=q) for a,q in v] for k,v in templates.items()}
 allq=[]
 for split in ['forget','retain']:
  f=rows(DATA/f'eval_data/{split}_5/train-00000-of-00001.parquet')
  for _,row in f.iterrows():
   for q in row['image understanding']:allq.append((q['Question'],split,q['Image']))
 unique=sorted(set(q for q,_,_ in allq)); stop=set('a an the is are does do what which how of in on at to or and with this that there any have has person persons photo image picture it its be'.split())
 def tokens(s):return set(normalize(s).split())-stop
 def sim(a,b):
  x,y=tokens(a),tokens(b);return len(x&y)/max(1,len(x|y))
 checks=[]
 for group,items in templates.items():
  for item in items:
   near=sorted([(sim(item['question'],q),q) for q in unique],reverse=True)[:3]
   checks.append(dict(group=group,**item,nearest=[dict(jaccard=s,question=q) for s,q in near],exact_overlap=normalize(item['question']) in {normalize(q) for q in unique},semantic_review='Distinct task: frame geometry/roll for B2; rendering/overlay/artifact properties for holdout. No identity, apparel, hair, emotion, gaze, background, or illumination queries.'))
 assert all(not c['exact_overlap'] and c['nearest'][0]['jaccard']<.5 for c in checks),checks
 cross=[dict(b2=a['attribute'],holdout=b['attribute'],jaccard=sim(a['question'],b['question'])) for a in templates['B2'] for b in templates['appearance_holdout']]
 atomic_json(OUT/'templates.json',dict(**templates,rejected_example_attributes=['hair colour/style','eyeglasses','clothing','background','expression','indoor/outdoor','headwear','jewellery','lighting','gaze'],reason='v3 illustrative attributes overlap evaluation tasks; replaced before any new candidate result',limitation='Heldout tasks diagnose imaging/appearance consistency; category diversity and reference refusal will be reported, not assumed.'))
 atomic_json(OUT/'template_similarity.json',dict(method='normalized content-word Jaccard; reject >=0.5; exact match rejection; semantic manual review of all unique face questions and nearest image-understanding questions',eval_question_count=len(allq),unique_question_count=len(unique),checks=checks,cross_group_checks=cross,semantic_overlap=False))
 # Freeze the future B2r image IDs now as well, without reading their outputs.
 atomic_json(OUT/'b2r_ids.json',dict(seed=SEED,ids=sorted(random.Random(SEED).sample(part['retain']['X_pool'],50))))
 now=datetime.now(timezone.utc).isoformat()
 protocol=dict(frozen_at=now,seed=SEED,split_sha256=sha256(ROOT/'wp1_review/fixed_person_split.json'),design_sha256=sha256(ROOT/'SMFA_WP1_experiment_design_v3.md'),delta_combined_pp=5,channel_upper_limits_pp={'image_memory':10,'text_memory':10},text_monitor_pp=5,soft_tie_rouge=.01,stop_improvement_rouge=.01,bootstrap_replicates=20000,bootstrap_seed=SEED,tau=.9,initial_margin_pp=0,
 margin_rule='After A only: h=max(central90 combined-memory halfwidths of two prescribed pairs). m=min(5,ceil(2*max(0,h-2.5))/2) pp; apply as normalized ROUGE-residual buffer. If h>=10, emphasize tradeoff and report noninferiority feasibility as poor. Never change delta. Buffer is heuristic, not a factuality-to-ROUGE conversion.',
 interpolation='Piecewise linear in observed combined forget-memory ROUGE; sort by x; same x retain best retain-image-memory y. Nonmonotonic segments retained. No extrapolation. Outside C1 range add c=0 if above max forgetting residual, then c=1.25,1.5,2,3 if below min, in sequence until bracketed; if still outside report unidentifiable and do not force a winner. Lambda ties within 1e-8 choose smaller lambda.',
 lambda_selection='Max C2 retain-image-memory ROUGE minus interpolated C1 at same F1 combined-memory ROUGE. All 3 must be bracketed or explicitly mark selection incomplete.',
 close_pair='C0 k10/20 vs fresh SMFA; C2 whose F1 combined-memory ROUGE is closest to SMFA vs fresh SMFA (distance ties smaller lambda).',
 judgment='Only after all A generations: interleave 975 judgments across three models, same qwen-plus settings/prompt as step0. Base/Original historical scores for A planning only; rejudge them in confirmation.',
 folds='Default twofold. A only F1+retain-dev. Fold2 independently repeats selection; reuse retain-dev and add F2 475/candidate; new lambda/operator requires subsequent new full development candidates.',
 stop_rule='At matched target and satisfying text monitor, C4a/C4/C5 all improve retain-image-memory ROUGE by <=0.01 over best eligible C0/C1/S0: stop parameter expansion; descriptive development rule, not statistical proof of dominance.',
 checks='C1 c1 six classes exactly saved MFA; down/vision exactly saved SMFA. Operator tests then 2layer/10sample pilot before full B1 statistics or C2/C3 generation. No new candidate results inspected before this freeze.')
 atomic_json(OUT/'protocol.json',protocol)
 md=f'''# WP1 A 阶段开工前冻结记录\n\n日期（UTC）：{now}\n\n此记录在任何新候选生成之前写定。人物划分沿用已冻结文件，不重抽。\n\n- 合并记忆采用单侧非劣效：中心 90% 区间上界严格小于 5 个百分点；图像、文本单项上界分别不超过 10 个百分点。10 是防止合并指标掩盖单项恶化的护栏，不等于两项都通过 5 点非劣效。\n- 同意 v3 四项新增：同期打分及漂移检查；开发文本残余监控阈值 +5 点；A 后只决定一次余量；第二折独立选参及必要补跑。\n- §6.2：匹配遗忘目标后，保留图像记忆 ROUGE 绝对差 <0.01 选软收缩；等于 0.01 时选实测较高者。0.01 与事实性 0.1 不作固定换算。\n- §6.4：在同一遗忘目标、满足文本监控的可比配置中，C4a/C4/C5 均未比 C0/C1/S0 最佳值提高超过 0.01，则停止扩参。这是开发阶段停止规则，不称为统计学支配证明；无法插值的配置不当作失败。\n- 主终点：确认保留集图像记忆事实性高于 SMFA，人物配对 95% 区间下界 >0。次终点：保留文本记忆、人脸理解及有意义性点估计下降 ≤0.2（各自评分尺度），报告区间；眼科拒答下降须事实性不降；遗忘眼科只报告。\n- 预先指定的两组重估：SMFA 对 C0 k10/20；SMFA 对 F1 合并记忆 ROUGE 最接近 SMFA 的 C2（距离相同取较小 λ）。取两个合并指标 h 中较大值规划。\n- bootstrap：人物联合重抽四模型，20,000 次，种子 {SEED}，每次重算均值之比分母；不截断。A 规划分母暂用历史 Base/Original，确认时同期重打。\n- 余量初值 0；A 重估后、B 插值前一次性定 m=min(5,ceil(2·max(0,h−2.5))/2) 点。m 用于归一化 ROUGE 残余目标的缓冲，不能保证事实性门槛通过；δ 始终 5。h≥10 时重点报告权衡及可行性不足。\n- 插值、λ 并列与超范围补点规则见 protocol.json；禁止外推。\n- 默认两折。当前 A 仅 F1 25 人 + 开发保留 100 人，2375 题，不生成或选用 F2/测试保留的新候选表现。未来第二折独立选参。\n- B1 固定种子 {SEED}，按原始比例最大余数分配：人脸64、眼科6、纯文本58。全部候选共享这128条。\n- B2 八项构图任务；留出六项成像外观任务；具体文字、相似度与语义复核存档。示例属性因与评测重叠而剔除。这是已知图上的辅助一致度诊断，不是有真值的正确率。\n- 合并精度继承通过自检的 PEFT 因子：multi FP32，text/RA FP16 后转 FP32；Original 先按官方加载为 FP16，更新与加法 FP32，最后转 FP16；不在前一候选上减回。\n- 启动门槛：C1 逐元素核对通过才评测 C0/C1；算子单测、2层10条试跑通过才全量统计与 C2/C3。试跑耗时/显存属于事后实测，不在此预填。\n\n冻结文件清单及哈希见 FREEZE.sha256。训练数据、图片、权重和大型统计不上传 Git。\n'''
 (OUT/'开工前冻结记录.md').write_text(md)
 paths=[OUT/name for name in ['freeze.py','protocol.json','b1_selection.json','templates.json','template_similarity.json','b2r_ids.json','开工前冻结记录.md']]
 (OUT/'FREEZE.sha256').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in paths))
 print(json.dumps(dict(frozen_at=now,quotas=quota,selection_people=len({x['person_id'] for x in selected}),freeze_sha256=sha256(OUT/'FREEZE.sha256'),max_template_jaccard=max(c['nearest'][0]['jaccard'] for c in checks)),indent=2))
if __name__=='__main__':main()
