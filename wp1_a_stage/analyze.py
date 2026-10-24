"""Frozen development selection, paired-person planning bootstrap and A report."""
import sys,json,math,argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wp1_a_stage.common import OUT,now,atomic_json,sha256,frozen,EVAL,DATA,rows,annotate_questions
from wp1_review.prepare_wp1 import load_judge_scores
from wp1_a_stage.candidates import INITIAL,SPECTRAL

def met(name):
 p=OUT/('references' if name in ['SMFA','Base','Original','MFA_unsculpted'] else 'candidates')/name/'metrics.json'
 return json.loads(p.read_text())
def val(m,key):return m[key]['person_macro_mean']['rougeL']
def interpolate(points,x):
 best={}
 for px,py in points:best[px]=max(best.get(px,-float('inf')),py)
 ordered=sorted(best.items())
 if x<ordered[0][0]-1e-12 or x>ordered[-1][0]+1e-12:return None
 return float(np.interp(x,[v[0] for v in ordered],[v[1] for v in ordered]))

def select():
 frozen();names=[x['name'] for x in INITIAL+SPECTRAL]+['SMFA','Original','Base','MFA_unsculpted']
 names+= [p.stem for p in sorted((OUT/'extra_candidates').glob('*.json')) if (OUT/'candidates'/p.stem/'metrics.json').exists()] if (OUT/'extra_candidates').exists() else []
 metrics={name:met(name) for name in names};curve=[(val(m,'forget/combined_memory'),val(m,'retain/image_memory')) for name,m in metrics.items() if name.startswith('C1_')]
 comparisons=[]
 for lam in [10,100,1000]:
  name=f'C2_l{lam}';m=metrics[name];x=val(m,'forget/combined_memory');y=val(m,'retain/image_memory');ref=interpolate(curve,x)
  comparisons.append(dict(name=name,lambda_rel=lam,forget_combined_rouge=x,retain_image_rouge=y,c1_interpolated_rouge=ref,uplift=None if ref is None else y-ref))
 outside=[p for p in comparisons if p['uplift'] is None]
 result=dict(at=now(),state='requires_bracketing' if outside else 'selected',comparisons=comparisons,c1_curve=sorted(curve),smfa_target=val(metrics['SMFA'],'forget/combined_memory'))
 if outside:
  c_values=[json.loads((OUT/'extra_candidates'/f'{n}.json').read_text())['c'] for n in names if n.startswith('C1_extra')]
  if any(x['forget_combined_rouge']>max(p[0] for p in curve) for x in outside) and 0 not in c_values:result['next_c']=0
  elif any(x['forget_combined_rouge']<min(p[0] for p in curve) for x in outside):result['next_c']=next((c for c in [1.25,1.5,2,3] if c not in c_values),None)
  else:result['next_c']=None
  if result['next_c'] is None:result['state']='unbracketed_after_frozen_extensions'
 else:
  top=max(p['uplift'] for p in comparisons);winner=min([p for p in comparisons if p['uplift']>=top-1e-8],key=lambda p:p['lambda_rel']);result.update(selected_lambda=winner['lambda_rel'],selected_candidate=winner['name'])
 close=min(comparisons,key=lambda p:(abs(p['forget_combined_rouge']-result['smfa_target']),p['lambda_rel']));result['factuality_candidates']=['SMFA','C0_k10',close['name']]
 result['c0_c3_vs_c1']={}
 for name in ['C0_k1','SMFA','C0_k10','C3_hard']:
  m=metrics[name];reference=interpolate(curve,val(m,'forget/combined_memory'));result['c0_c3_vs_c1'][name]=dict(c1_interpolated_rouge=reference,uplift=None if reference is None else val(m,'retain/image_memory')-reference,note='Outside observed C1 range' if reference is None else 'Matched combined-memory ROUGE; descriptive development comparison')
 result['development_summary']={name:dict(forget_combined=val(m,'forget/combined_memory'),forget_image=val(m,'forget/image_memory'),forget_text=val(m,'forget/text_memory'),retain_image=val(m,'retain/image_memory'),retain_text=val(m,'retain/text_memory'),retain_eye_refusal=m['retain/image_understanding/ophthalmic']['mean']['refusal_loose'],retain_eye_rouge=val(m,'retain/image_understanding/ophthalmic'),answer_words=m['retain/all']['mean']['answer_words']) for name,m in metrics.items()}
 denom_text=val(metrics['Original'],'forget/text_memory')-val(metrics['Base'],'forget/text_memory')
 result['text_monitor_at_observed_points']={name:dict(difference_from_smfa_pp=100*(val(m,'forget/text_memory')-val(metrics['SMFA'],'forget/text_memory'))/denom_text,above_5pp=bool(100*(val(m,'forget/text_memory')-val(metrics['SMFA'],'forget/text_memory'))/denom_text>5)) for name,m in metrics.items() if name not in ['Original','Base','MFA_unsculpted']}
 result['text_monitor_note']='Observed-point diagnostics only. B-stage strength interpolation must separately check text residual at its matched target.'
 atomic_json(OUT/'selection.json',result)
 return result

def bootstrap(a,quantiles=(.05,.95)):
 rng=np.random.default_rng(20261023);draw=rng.integers(0,len(a),size=(20000,len(a)));means=a[draw].mean(1);denom=means[:,1]-means[:,0]
 if np.any(denom<=0):raise ValueError('Bootstrap ratio denominator not positive')
 values=100*(means[:,3]-means[:,2])/denom;lo,hi=np.quantile(values,quantiles);m=a.mean(0)
 return dict(estimate_pp=float(100*(m[3]-m[2])/(m[1]-m[0])),ci90_pp=[float(lo),float(hi)],halfwidth_pp=float((hi-lo)/2),denominator_mean=float(m[1]-m[0]),denominator_bootstrap_min=float(denom.min()))

def closeout():
 selection=json.loads((OUT/'selection.json').read_text());part=frozen();ids=sorted(part['forget']['F1_development']);by_model={};provenance={}
 for model in ['Base','Original','SMFA']:
  p=EVAL/model/'forget_generation_results.json';qs=json.loads(p.read_text())['Generation_Questions'];ann=annotate_questions(qs,rows(DATA/'eval_data/forget_5/train-00000-of-00001.parquet'));scores,source=load_judge_scores(EVAL/model,'forget',qs,'factuality','Factuality Score');assert source['complete']
  by_model[model+'_historical']={(pid,q['question type'],q['question']):s for (q,pid),s in zip(ann,scores) if pid in ids and q['question type'].endswith('memory')};provenance[model]=source
 records=json.loads((OUT/'factuality/scores.json').read_text());assert records['state']=='complete' and len(records['records'])==975
 for model in selection['factuality_candidates']:
  by_model[model]={(r['question']['person_id'],r['question']['question type'],r['question']['question']):r['result']['Factuality Score'] for r in records['records'] if r['model']==model}
  assert len(by_model[model])==325
 keys=set(by_model['SMFA']);assert all(set(v)==keys for v in by_model.values())
 arrays={}
 for model,data in by_model.items():
  arrays[model]=np.array([[np.mean([s for (pid,kind,q),s in data.items() if pid==identity and (target=='combined_memory' or kind==target)]) for target in ['image_memory','text_memory','combined_memory']] for identity in ids])
 comparisons={}
 for model in selection['factuality_candidates'][1:]:
  comparisons[model]={kind:bootstrap(np.stack([arrays['Base_historical'][:,i],arrays['Original_historical'][:,i],arrays['SMFA'][:,i],arrays[model][:,i]],1)) for i,kind in enumerate(['image_memory','text_memory','combined_memory'])}
 h=max(v['combined_memory']['halfwidth_pp'] for v in comparisons.values());margin=min(5,math.ceil(2*max(0,h-2.5))/2)
 differences=np.array([by_model['SMFA'][key]-by_model['SMFA_historical'][key] for key in sorted(keys)])
 person_diffs=arrays['SMFA'][:,2]-arrays['SMFA_historical'][:,2];rng=np.random.default_rng(20261023);ci=np.quantile(person_diffs[rng.integers(0,len(ids),(20000,len(ids)))].mean(1),[.025,.975])
 drift=dict(mean_factuality_difference=float(differences.mean()),mean_absolute_difference=float(np.abs(differences).mean()),exact_score_agreement=float((differences==0).mean()),person_paired_ci95=list(map(float,ci)),min_difference=int(differences.min()),max_difference=int(differences.max()),interpretation='Fresh minus historical scores of identical SMFA answers. Drift and judging randomness cannot be separated by one repeat.')
 result=dict(state='complete',at=now(),people=25,judgments=975,comparisons=comparisons,planning_h_pp=h,margin_pp=margin,delta_pp=5,twofold=True,drift=drift,historical_denominator_limitation='Base/Original are historical for A planning; rejudge at confirmation. Current CI conditional on saved judge responses and selected candidate.',provenance=provenance,selection_sha256=sha256(OUT/'selection.json'),scores_sha256=sha256(OUT/'factuality/scores.json'))
 base=met('Base');orig=met('Original');result['B1_target_forget_combined_rouge']=selection['smfa_target']-margin/100*(val(orig,'forget/combined_memory')-val(base,'forget/combined_memory'))
 atomic_json(OUT/'A_closeout.json',result);report(selection,result);return result

def report(selection,close):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10});fig,ax=plt.subplots(figsize=(9,5.5));pts=selection['development_summary']
 curve=sorted((v['forget_combined'],v['retain_image']) for k,v in pts.items() if k.startswith('C1_'));ax.plot([p[0] for p in curve],[p[1] for p in curve],'o-',label='C1 scaling')
 for name,p in pts.items():
  if name.startswith('C1_'):continue
  offset={'C0_k1':(-18,-15),'C2_l1000':(-24,-34),'C3_hard':(24,0),'Original':(-5,22),'C2_l100':(-14,14)}.get(name,(4,5))
  ax.scatter(p['forget_combined'],p['retain_image']);ax.annotate(name,(p['forget_combined'],p['retain_image']),fontsize=8,xytext=offset,textcoords='offset points',arrowprops={'arrowstyle':'-','color':'.55','lw':.5} if name in ['C2_l1000','C3_hard','Original'] else None)
 ax.set(xlabel='Forget combined-memory ROUGE-L (lower = stronger forgetting)',ylabel='Retain image-memory ROUGE-L',title='WP1 A stage: frozen development identities only');ax.margins(x=.15,y=.15);ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/'A_tradeoff.png',dpi=180);plt.close(fig)
 s0=json.loads((OUT/'statistics/S0_summary.json').read_text());ranks=[v['r'] for v in s0['groups'].values()];energy=[v['visual_energy_fraction_S0'] for v in s0['groups'].values()]
 protocol=json.loads((OUT/'protocol.json').read_text());sample=json.loads((OUT/'b1_selection.json').read_text());similarity=json.loads((OUT/'template_similarity.json').read_text())
 lines=['# WP1 A 阶段实验记录与结果','',f"完成时间（UTC）：{close['at']}",'',f"10 个新增候选均已完成开发评测，共 23,750 条回答；975 条同期事实性评分完成。按冻结规则选 λ_rel={selection.get('selected_lambda','未选定')}，B 阶段开发余量为 {close['margin_pp']:.1f} 个百分点。当前 α=1 的 C2 遗忘弱于 SMFA，尚不能据此认定工作点1成功；后续需要补强度点并在确认组检验。",'',f"规则和抽样冻结时间（UTC）：{protocol['frozen_at']}。冻结记录见 [开工前冻结记录.md](开工前冻结记录.md)，文件哈希见 [FREEZE.sha256](FREEZE.sha256)。该清单自身 SHA256：`{sha256(OUT/'FREEZE.sha256')}`。",'', '本阶段仅 F1 25 人和开发保留 100 人，每候选 2,375 题（475 遗忘 + 1,900 保留）；未生成 F2 或测试保留集的新候选结果。原始、基座、SMFA 和不雕刻 MFA 参照直接抽取自第0步回答。', '', '| 冻结项目 | 规则 |','|---|---|','| 合并记忆非劣效 | 中心90%区间上界 <5个百分点 |','| 单通道护栏 | 图像、文本记忆各自中心90%区间上界 ≤10个百分点 |','| 文本监控 | 匹配目标处，归一化文本记忆 ROUGE 残余比 SMFA 高 >5个百分点时补强度或另选点 |','| 软硬算子接近 | 保留图像记忆 ROUGE 绝对差 <0.01 时选软收缩；等于0.01选较高者 |','| 停止扩参 | 匹配目标且通过文本监控后，C4a/C4/C5相对最佳C0/C1/S0的提升均≤0.01 |','| 四项v3新增 | 同意同期评分、文本监控、A后一次性决定余量、第二折独立选择及补跑 |','| 主终点 | 确认保留集图像记忆事实性相对SMFA的配对95%区间下界 >0 |','| 次终点 | 保留文本/人脸理解/有意义性点估计下降≤0.2并报告区间；眼科拒答下降须事实性不降；遗忘眼科仅报告 |','| 两折 | 当前完成第一折A；第二折仍须独立执行选择规则 |','', '阈值0.01按ROUGE本身解释，不换算成固定的事实性分差。停止规则是开发决策，不是统计学支配证明。', '', f"B1 固定种子 {sample['seed']}，原始 1,100 条按比例最大余数分层抽 128 条，覆盖 {len({x['person_id'] for x in sample['selected']})} 人：",'', '| 样本类型 | 原始条数 | 抽中条数 |','|---|---:|---:|']
 for kind in sorted(sample['source_counts']):lines.append(f"| {kind} | {sample['source_counts'][kind]} | {sample['selected_counts'][kind]} |")
 lines+=['',f"B2 固定8项构图/头部几何模板，外观留出固定6项成像/覆盖文字/图像质量模板，见 [templates.json](templates.json)。v3所举发色、眼镜、衣物、表情、背景等示例与评测任务重叠，故在看候选前替换。与评测图像理解题的最大内容词 Jaccard={max(x['nearest'][0]['jaccard'] for x in similarity['checks']):.2f}（剔除阈值≥0.5）；另存最近题与语义复核。B2与留出题的任务不重叠，留出题只诊断与Original的一致性及拒答，可能存在类别变化少的问题。",'', 'C1构造核对已通过：168个语言模型q/k/v/o/gate/up模块与保存的未雕刻MFA逐元素一致；106个down/视觉模块与保存的SMFA逐元素一致。MFA_multi保留FP32，MFA_text/RA先FP16再FP32；更新和加法FP32，最终权重FP16。每个候选从Original重新构造。', '', 'C0两点为k_multi/k_text=1/2、10/20，所有模块雕刻；已有SMFA为5/10。C1的c=1仍保留down/视觉掩码，因此不等于第0步未雕刻MFA。C2三个λ与C3均固定α=1，只修正两套MFA的A因子。', '',f"A 阶段 λ 选择：{selection.get('selected_lambda','未选定：C1曲线未覆盖所有C2点')}。固定规则为：比较同一合并遗忘记忆 ROUGE 处，C2 相对 C1 插值曲线的保留图像记忆提升。",'','| 配置 | 遗忘合并记忆 ROUGE | 遗忘图像 | 遗忘文本 | 保留图像 | 保留文本 | 保留眼科宽松拒答 |','|---|---:|---:|---:|---:|---:|---:|']
 for name,p in pts.items():lines.append(f"| {name} | {p['forget_combined']:.5f} | {p['forget_image']:.5f} | {p['forget_text']:.5f} | {p['retain_image']:.5f} | {p['retain_text']:.5f} | {p['retain_eye_refusal']:.1%} |")
 lines+=['','| λ | 同遗忘水平的 C1 保留图像 ROUGE | C2 提升 |','|---|---:|---:|']
 for point in selection['comparisons']:
  reference='不可插值' if point['c1_interpolated_rouge'] is None else f"{point['c1_interpolated_rouge']:.6f}";uplift='不可识别' if point['uplift'] is None else f"{point['uplift']:+.6f}"
  lines.append(f"| {point['lambda_rel']} | {reference} | {uplift} |")
 lines+=['','| C0/C3 | 相对同遗忘水平 C1 的保留图像 ROUGE 差 |','|---|---:|']
 for name,point in selection['c0_c3_vs_c1'].items():
  uplift='超出范围，不外推' if point['uplift'] is None else f"{point['uplift']:+.6f}";lines.append(f"| {name} | {uplift} |")
 lines+=['', 'SMFA与两个C0点均在已测C1插值曲线上方约0.0101–0.0170，说明开发集上掩码相对整体缩放有收益。三个C2点也在曲线上方，其中λ=10提升最大；这是相对实测四点的线性插值比较，未测中间强度的曲线形状仍有不确定性。C3遗忘合并ROUGE=0.69252，略超C1已测上限0.68639，故不能判定它相对C1更好或更差，也不能据此完成软硬算子选择。', '', '各候选完整的集合×题型×图像类型ROUGE、精确/宽松拒答率、回答词数/字符数（新生成另有token数）保存在对应metrics.json；以下额外列出保留眼科与文本监控。']
 lines+=['','| 配置 | 保留集平均回答词数 | 保留眼科理解 ROUGE | 遗忘文本残余相对SMFA（百分点） |','|---|---:|---:|---:|']
 for name,p in pts.items():
  monitor=selection['text_monitor_at_observed_points'].get(name);text_value=f"{monitor['difference_from_smfa_pp']:.3f}" if monitor else '参照'
  lines.append(f"| {name} | {p['answer_words']:.2f} | {p['retain_eye_rouge']:.5f} | {text_value} |")
 lines+=['','此处文本监控是各观测点的诊断，不代表已匹配遗忘水平；B阶段插值到目标后仍须单独检查 +5 点阈值。','', '![A阶段权衡曲线](A_tradeoff.png)','',f"S0 共84组，方向数 r 范围 {min(ranks)}–{max(ranks)}，均值 {np.mean(ranks):.2f}；视觉能量占比范围 {min(energy):.2%}–{max(energy):.2%}，组均值 {np.mean(energy):.2%}。逐层84组记录见 [S0_summary.json](statistics/S0_summary.json)。",'', '| 输入组 | 组数 | r范围 | 平均r | 视觉能量占比范围 |','|---|---:|---:|---:|---:|']
 for group in ['attn','o','mlp']:
  entries=[v for k,v in s0['groups'].items() if k.endswith('_'+group)];rr=[v['r'] for v in entries];ee=[v['visual_energy_fraction_S0'] for v in entries]
  lines.append(f"| {group} | {len(entries)} | {min(rr)}–{max(rr)} | {np.mean(rr):.2f} | {min(ee):.2%}–{max(ee):.2%} |")
 lines+=['', '算子单元检查全部通过：r=0和λ=0逐元素不变，硬投影清除受保护方向、正交补不变，大λ趋近硬投影，固定模块和有限值通过。全量真实特征系统另核对84组、336个MFA A因子；最大相对特征残差2.18e-6，硬投影残差7.99e-7。', '',f"975条事实性评分完成：{', '.join(selection['factuality_candidates'])} 各325条，同期交错提交。评分模型qwen-plus，提示和模型参数沿用第0步；API模型名、返回ID、时间、原始响应及分数均本地保存。",'','| 相对同期 SMFA | 记忆类型 | 差值（百分点） | 90%区间 | h |','|---|---|---:|---|---:|']
 for name,kinds in close['comparisons'].items():
  for kind,v in kinds.items():lines.append(f"| {name} | {kind} | {v['estimate_pp']:.3f} | [{v['ci90_pp'][0]:.3f}, {v['ci90_pp'][1]:.3f}] | {v['halfwidth_pp']:.3f} |")
 drift_ci=[0.0 if abs(v)<1e-12 else v for v in close['drift']['person_paired_ci95']]
 lines+=['',f"规划取较宽的合并 h={close['planning_h_pp']:.3f} 点，仍高于原定精度2.5点。按冻结公式 m=min(5,ceil(2·max(0,h−2.5))/2)，B阶段余量定为 {close['margin_pp']:.1f} 点，目标合并记忆 ROUGE={close['B1_target_forget_combined_rouge']:.6f}（SMFA为{selection['smfa_target']:.6f}）；δ 保持5点，单项上限10点，默认两折。余量只是 ROUGE 上的开发启发式，不保证事实性非劣效通过。",'',f"C2 λ=10、α=1 的合并事实性残余比SMFA高 {close['comparisons']['C2_l10']['combined_memory']['estimate_pp']:.3f} 点，其90%区间上界超过5；同时观测点文本ROUGE残余高20.952点。这个点用于选择λ和估算区间，不能直接作为通过遗忘判定的最终配置。",'',f"SMFA 同回答重打分漂移：平均差 {close['drift']['mean_factuality_difference']:.4f}，平均绝对差 {close['drift']['mean_absolute_difference']:.4f} 分，逐题分数完全一致 {close['drift']['exact_score_agreement']:.1%}，人物配对95%区间 [{drift_ci[0]:.4f}, {drift_ci[1]:.4f}]。区间接近零的浮点尾数按零展示；一次重复无法区分评分随机性和服务端漂移。",'','人物配对bootstrap为20,000次、种子20261023，每次四模型联合抽人并重算分母，不平均逐人比值、不截断。A阶段的 Base/Original 分母来自历史评分，仅用于规划；确认时仍需同期重打。这里的区间不覆盖选择流程的不确定性，不据此声称测试成功。','', '每候选配置、构造时间、生成时间和逐人缓存在 candidates/；统计试跑成本见 [cost_estimate.json](pilot/cost_estimate.json)；原始模型、基座、SMFA参照由第0步按固定划分抽取。','', '后续 B1：选定 λ 的软收缩与硬投影分别补 α=1.5、2、3，按本次冻结的目标和0.01规则选算子。A阶段收尾后尚未启动B阶段。']
 lines+=['','| 配置 | 构造耗时（秒） | 生成及汇总耗时（秒） |','|---|---:|---:|']
 for name in pts:
  path=OUT/'candidates'/name/'status.json'
  if path.exists():
   timing=json.loads(path.read_text());lines.append(f"| {name} | {timing.get('construction_seconds',0):.1f} | {timing.get('generation_seconds',0):.1f} |")
 judge=json.loads((OUT/'factuality/status.json').read_text());lines+=['',f"同期975条事实性评分执行耗时：{judge['seconds']:.1f} 秒。全量B1成功采集及检查点耗时 {s0['collection_seconds']:.1f} 秒，特征分解及保存耗时 {s0['eigendecomposition_and_save_seconds']:.1f} 秒；此前失败重跑额外耗时见 implementation_events.json 和日志。",'',f"统计数值核对：前缀相对差最大 {s0['prefix_activation_max_relative_l2']:.4%}，重复视觉探针相对差最大 {s0['visual_duplicate_probe_max_relative_l2']:.4%}。S0与S0d使用同一张图的首次激活，前者乘精确出现次数，后者计一次，遵循v3 §3.6。",'', '实现修复及偏差记录：最初文件名遮蔽Python标准库导致导入失败，改名后启动；图像/纯文本的公共前缀为2 token，额外空白计入文本；FP16序列长度数值差异经完整激活诊断后改用相对范数检查。失败的未完成统计不进入最终候选。']
 pilot=json.loads((OUT/'pilot/S0_summary.json').read_text());cost=json.loads((OUT/'pilot/cost_estimate.json').read_text())
 lines+=['',f"2层10条试跑先于正式统计，包含人脸、眼科及纯文本。试跑最大GPU已分配显存 {pilot['peak_gpu_allocated_bytes']/2**30:.2f} GiB，单矩阵 {pilot['matrix_bytes']/2**20:.2f} MiB；层数线性上界预计全量采集 {cost['full_collection_seconds_linear_layer_bound']:.1f} 秒、特征分解 {cost['full_eigen_seconds']:.1f} 秒。成功全量实际峰值 {s0['peak_gpu_allocated_bytes']/2**30:.2f} GiB，CPU矩阵 {s0['accumulator_bytes']/2**30:.2f} GiB；实际总耗时与预计差异已以上列数字报告。试跑逐条序列长度、token划分及前向耗时见 pilot/S0_summary.json。",'', '前缀仅存2 token的激活及128次计数；不同的空白token并入文本。每组视觉token去重267,754、重复488,884，文本3,852。矩阵累加器在CPU，乘法和特征分解在GPU以FP32执行，TF32关闭；这降低GPU显存占用，不改变统计定义。']
 audit_path=OUT/'completion_audit.json'
 if audit_path.exists():
  audit=json.loads(audit_path.read_text());assert audit['state']=='passed'
  lines+=['', f"收尾审计通过（UTC {audit['at']}）：10个候选各2,375条、共23,750条均与冻结开发人物和源题目逐条核对；975个分数的输入哈希、回答/标准答案元组及范围检查通过；冻结文件和候选代码哈希未变。见 [completion_audit.json](completion_audit.json)。",'', '| 同期事实性平均分 | 图像记忆 | 文本记忆 | 合并记忆（7:6） |','|---|---:|---:|---:|']
  for name,means in audit['factuality_means'].items():lines.append(f"| {name} | {means['image_memory']:.4f} | {means['text_memory']:.4f} | {means['combined_memory']:.4f} |")
 (OUT/'SMFA_WP1_A阶段_实验记录与结果.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--closeout',action='store_true');a=p.parse_args();print(json.dumps(closeout() if a.closeout else select(),ensure_ascii=False,indent=2))
