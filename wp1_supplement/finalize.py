"""Completion audit and aggregate-only Chinese experiment report."""
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from wp1_a_stage.common import dev_frames,questions_for_row
from wp1_supplement.common import OUT,PARENT,now,atomic_json,sha256,frozen,summary

def read(path):return json.loads(path.read_text())

def fmt(x,n=6):return '—' if x is None else f'{x:.{n}f}'

def audit():
    protocol,partition=frozen();frozen_time=datetime.fromisoformat(protocol['frozen_at'])
    # Verify that the original result inventory itself and all its contents are intact.
    original_results=0
    for line in (PARENT/'RESULTS.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);assert sha256(PARENT/name)==digest,('Original result changed',name);original_results+=1
    frames=dev_frames();expected={split:{row.ID:questions_for_row(row,split,index) for index,row in frame.iterrows()} for split,frame in frames.items()}
    candidates={};earliest=None;generation_times=[]
    b1=read(OUT/'B1_selection_inputs.json');assert sha256(OUT/'c1_selection.json')==b1['c1_selection_sha256']
    correction=read(OUT/'operator_comparison_correction.json')
    assert (OUT/'OPERATOR_COMPARISON.sha256').read_text()==sha256(OUT/'operator_comparison_correction.json')+'  operator_comparison_correction.json\n'
    expected_b1=sha256(OUT/'B1_selection_inputs.json')+'  B1_selection_inputs.json\n'
    assert (OUT/'B1_selection_inputs.sha256').read_text()==expected_b1
    for path in sorted((OUT/'candidates').glob('*/manifest.json')):
        directory=path.parent;manifest=read(path);status=read(directory/'status.json');name=directory.name
        assert status['state']=='complete',('Incomplete candidate',name)
        cfg=manifest['config'];assert manifest['freeze_sha256']==sha256(OUT/'FREEZE_supplement.sha256')
        for file,digest in manifest['code_sha256'].items():assert sha256(OUT.parent/file)==digest,('Worker source changed',file)
        assert datetime.fromisoformat(status['generation_started_at'])>frozen_time
        if cfg['family'] in ('C2','C3'):
            assert datetime.fromisoformat(status['generation_started_at'])>datetime.fromisoformat(b1['frozen_at'])
            assert datetime.fromisoformat(status['generation_started_at'])>datetime.fromisoformat(correction['frozen_at'])
        n=0;per_question_seconds=0.
        for split in ('forget','retain'):
            qs=read(directory/f'{split}_generation_results.json')['Generation_Questions']
            assert len(qs)==19*len(expected[split])
            grouped={}
            for q in qs:
                assert q['person_id'] in expected[split] and q['split']==split
                grouped.setdefault(q['person_id'],[]).append(q)
                timestamp=datetime.fromisoformat(q['generated_at']);assert timestamp>frozen_time
                generation_times.append(timestamp);earliest=timestamp if earliest is None else min(earliest,timestamp)
                per_question_seconds+=q['generation_seconds'];n+=1
            assert set(grouped)==set(expected[split])
            for pid,actual in grouped.items():
                assert len(actual)==19
                for q,ref in zip(actual,expected[split][pid]):assert all(q[k]==v for k,v in ref.items()),(name,pid)
        assert n==2375
        candidates[name]=dict(config=cfg,questions=n,gpu=status['gpu'],generation_started_at=status['generation_started_at'],
                              finished_at=status['finished_at'],generation_wall_seconds=status['generation_seconds'],
                              generation_sum_question_seconds=per_question_seconds,construction_seconds=status['construction_seconds'],
                              metrics_sha256=sha256(directory/'metrics.json'),generation_sha256={split:sha256(directory/f'{split}_generation_results.json') for split in ('forget','retain')})
    appearance=read(OUT/'appearance/summary.json')
    appearance_sets={}
    for path in sorted((OUT/'appearance').glob('*/*/generation_results.json')):
        qs=read(path)['questions'];assert len(qs)==150
        assert {q['person_id'] for q in qs}==set(partition['forget']['F1_development'])
        assert all(datetime.fromisoformat(q['generated_at'])>frozen_time for q in qs)
        first=min(datetime.fromisoformat(q['generated_at']) for q in qs)
        earliest=first if earliest is None else min(earliest,first)
        appearance_sets[str(path.parent.relative_to(OUT))]=dict(questions=150,sha256=sha256(path))
    diag=read(OUT/'diagnostics/status.json');assert diag['state']=='complete' and diag['layers']==list(range(28))
    for name,count in [('D_fm',325),('D_fi',125),('D_rh',300)]:
        s=read(OUT/'diagnostics'/name/'summary.json');assert s['state']=='complete' and s['samples']==count
        assert s['prompt_only'] and s['prefix_excluded'] and datetime.fromisoformat(s['started_at'])>frozen_time
        assert len(s['groups'])==84
    assert read(OUT/'diagnostics/pilot/validation.json')['state']=='passed'
    assert read(OUT/'diagnostics/unit_checks.json')['state']=='passed'
    workflow=read(OUT/'workflow_unit_checks.json');assert workflow['state']=='passed'
    for file,digest in workflow['code_sha256'].items():assert sha256(OUT/file)==digest
    scope=read(OUT/'diagnostics/scope_extension.json')
    assert datetime.fromisoformat(scope['at'])<datetime.fromisoformat(read(OUT/'diagnostics/D_fm/summary.json')['started_at'])
    optional_stability=OUT/'stability/status.json'
    if optional_stability.exists():
        stable=read(optional_stability);assert stable['state']=='complete'
        assert stable['code_sha256']==sha256(OUT/'stability.py')
        split=read(OUT/'stability/selection.json')
        assert split['seed']==20261023 and sorted(sum(split['halves'],[]))==list(range(128))
        assert all(len(h)==64 for h in split['halves'])
        assert (OUT/'stability/SELECTION.sha256').read_text()==sha256(OUT/'stability/selection.json')+'  selection.json\n'
        assert len(read(OUT/'stability/results.json')['groups'])==84
    bootstrap=read(OUT/'c1_bootstrap.json');assert bootstrap['seed']==20261023 and bootstrap['replicates']==20000
    c1=read(OUT/'c1_selection.json');assert c1['selected_lambda']==b1['selected_lambda']
    matches={family:read(OUT/'matching'/f'{family}.json') for family in ('C0','C1','C2','C3')}
    for family,m in matches.items():
        if m['state']=='matched':
            s=m['summary'];assert protocol['hit_range'][0]<=s['forget_combined']<=protocol['hit_range'][1]
            assert m['text_monitor_difference_pp']<=protocol['text_monitor_pp']
            assert (OUT/'candidates'/m['selected_config']['name']/'metrics.json').exists()
    operator=read(OUT/'B1_operator_selection.json')
    if operator['selected_family']:
        assert matches[operator['selected_family']]['state']=='matched'
    out=dict(state='passed',at=now(),frozen_at=protocol['frozen_at'],earliest_new_answer_at=earliest.isoformat(),
             freeze_precedes_every_new_answer=True,freeze_precedes_candidate_generation=True,
             parent_frozen_inputs_intact=True,parent_results_intact=True,parent_results_verified_files=original_results,
             supplement_freeze_sha256=sha256(OUT/'FREEZE_supplement.sha256'),b1_lambda_locked_before_new_alpha=True,
             n_new_candidates=len(candidates),n_candidate_questions=sum(c['questions'] for c in candidates.values()),
             candidates=candidates,appearance_sets=appearance_sets,active_appearance_bank=appearance['active_bank'],
             diagnostic_layers=28,diagnostic_groups=84,bootstrap_replicates=20000,bootstrap_seed=20261023,
             candidate_cohorts='F1_development 25 + retain development 100 only',no_F2_or_test_retain_candidates=True,
             no_API_scoring=True,matched_families=[family for family,m in matches.items() if m['state']=='matched'],
             unmatched_families=[family for family,m in matches.items() if m['state']!='matched'],
             selection_state=operator['state'],selected_operator=operator['selected_family'])
    atomic_json(OUT/'completion_audit.json',out);return out

def report(audit_result):
    protocol,_=frozen();eight=read(OUT/'c1_eight_point_selection.json');selected=read(OUT/'c1_selection.json')
    boot=read(OUT/'c1_bootstrap.json');operator=read(OUT/'B1_operator_selection.json')
    appearance=read(OUT/'appearance/summary.json');lines=[]
    lines+=['# A 补点与 B1：实验记录及结果','',f'完成报告生成时间（UTC）：{now()}','',
            f'补充冻结时间：{protocol["frozen_at"]}。最早新回答时间：{audit_result["earliest_new_answer_at"]}。冻结顺序检查通过。',
            f'冻结清单 SHA-256：`{sha256(OUT/"FREEZE_supplement.sha256")}`。原 A 阶段全部冻结文件和 {audit_result["parent_results_verified_files"]} 项结果哈希核验通过。','',
            '## 规则与范围','',
            f'- 目标合并记忆 ROUGE：{protocol["target"]:.10f}；实测命中范围 [0.44029, 0.45229]。',
            '- 沿用 m=2.0、δ=5、单通道上限10个百分点、文本监控+5个百分点、软硬差小于0.01选软收缩。',
            '- 候选仅用 F1 25 人和开发保留100人，每配置2,375题；无 F2/测试保留集候选生成，无 API 打分。',
            '- 新 λ 在任何 B1 新 α 生成前单独锁定。诊断结果不参与选参。',
            '- 保留冻结的十层诊断切片；在采集前另存确定性的十八层扩展记录，完整诊断覆盖全部28层、84组。',
            '- 不进入 B2 或第二折。另利用等待 C1 最后一个点的空档，做固定分半的 B1 文本稳定性诊断，不参与选参。','',
            '## C1 补点与 λ 重选','',
            '| c | 遗忘合并记忆 ROUGE | 保留图像记忆 ROUGE | 保留眼科拒答率 | 保留回答词数 |',
            '|---:|---:|---:|---:|---:|']
    for p in sorted(selected['c1_curve'],key=lambda p:p['parameter']):
        lines.append(f'| {p["parameter"]:.8g} | {p["forget_combined"]:.6f} | {p["retain_image"]:.6f} | {p["retain_eye_refusal"]:.2%} | {p["answer_words"]:.2f} |')
    lines+=['',f'原四点 λ={selected["original_selection"]["selected_lambda"]}；八点重选 λ={eight["selected_lambda"]}；必要夹点后的最终重选 λ={selected["selected_lambda"]}。',
            '', '![C1补点与B1实测曲线](figures/supplement_tradeoffs.png)', '',
            '图中散点为实测配置，连线仅为插值；阴影表示冻结的最终命中范围。','',
            '','| λ_rel | C2 遗忘合并 ROUGE | C2 保留图像 ROUGE | 新 C1 插值参照 | 超出参照 |','|---:|---:|---:|---:|---:|']
    for c in selected['comparisons']:
        lines.append(f'| {c["lambda_rel"]} | {c["forget_combined_rouge"]:.6f} | {c["retain_image_rouge"]:.6f} | {fmt(c["c1_interpolated_rouge"])} | {fmt(c["uplift"])} |')
    lines+=['','### 人物 bootstrap','',
            '独立重抽遗忘与保留人物，同一人群内各模型共用抽样；20,000次、种子20261023，每次重新插值，无外推。',
            '**以下区间不含曲率误差和完整选择流程的不确定性，也不是事实性分的区间。**','',
            '| 配置 | 相对 C1 的保留图像 ROUGE 差 | 条件95%区间 | 有效重抽次数 |','|---|---:|---|---:|']
    for name,c in boot['comparisons'].items():
        ci=c['conditional_ci95'];citext='—' if ci is None else f'[{ci[0]:.6f}, {ci[1]:.6f}]'
        lines.append(f'| {name} | {fmt(c["estimate"])} | {citext} | {c["valid_replicates"]} |')
    lines+=['','## 外观留出题灵敏度','',
            '正对照为未雕刻 MFA 相对 Original：拒答净增幅<2个百分点，且全部150题的平均ROUGE-L一致度≥0.90，判为不灵敏。SMFA作为辅助对比。Original拒答子集和非拒答子集另存。','',
            '| 模板组 | 对比模型 | 一致度 ROUGE-L | 拒答增幅（百分点） | 判不灵敏 |','|---|---|---:|---:|---|']
    for path in sorted((OUT/'appearance').glob('*/sensitivity.json')):
        s=read(path)
        for name,c in s['comparisons'].items():
            lines.append(f'| {s["bank"]} | {name} | {c["mean_agreement_rougeL"]:.6f} | {100*c["refusal_increment"]:.3f} | {c["insensitive"]} |')
    lines+=['',f'最终模板字段：`{appearance["active_bank"]}`；最终正对照仍不灵敏：{appearance["active_insensitive"]}。',
            '若主模板不灵敏，按预定规则整套切换一次，原结果保留并写协议修订；不按单题表现换题，不再搜索第三套模板。若备用也不灵敏，后续外观诊断接近零不能作为同图感知恢复无效的证据。','',
            '## 激活与 LoRA 诊断','',
            'Original 仅做提示前向，不放入参考答案；视觉按图去重，文本分带图/纯文本，并单列合并文本，公共前缀不计入诊断矩阵。S0、S0d、S1 的基底来自原 B1 统计。','',
            '| 数据集 | token | S0平均覆盖 | S0d平均覆盖 | S1平均覆盖 |','|---|---|---:|---:|---:|']
    coverage=read(OUT/'diagnostics/coverage.json')['rows']
    for dataset in ('D_fm','D_rh','D_fi'):
        for token in ('V_unique','T_image','T_pure','T_all'):
            vals=[]
            for spec in ('S0','S0d','S1'):
                a=[r['coverage'] for r in coverage if r['dataset']==dataset and r['token_type']==token and r['spectrum']==spec and r['coverage'] is not None]
                vals.append(float(np.mean(a)) if a else None)
            lines.append(f'| {dataset} | {token} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} |')
    lines+=['','上表为84组等权平均；逐组方向数、覆盖率、文本分型和原始迹均见 `diagnostics/coverage.json`。零token项为缺失，不填0。','',
            '| 数据集 | token | C2 λ10响应 | C2 λ100响应 | C2 λ1000响应 | C3硬投影响应 |','|---|---|---:|---:|---:|---:|']
    responses=read(OUT/'diagnostics/response_ratios.json')['rows']
    for dataset in ('D_fm','D_rh','D_fi'):
        for token in ('V_unique','T_image','T_pure','T_all'):
            vals=[]
            for candidate in ('C2_l10','C2_l100','C2_l1000','C3_hard'):
                a=[r for r in responses if r['dataset']==dataset and r['token_type']==token and r['candidate']==candidate]
                denominator=sum(r['raw_energy'] for r in a);numerator=sum(r['corrected_energy'] for r in a)
                vals.append((numerator/denominator)**.5 if denominator>0 else None)
            lines.append(f'| {dataset} | {token} | '+ ' | '.join(fmt(v) for v in vals)+' |')
    lines+=['','响应按所有六类模块累加能量后开方，不是响应比的算术平均；α=1，针对LoRA更新，不包括基座相加后的fp16舍入。分层/模块结果保存在 `response_ratios.json`。','']
    counts=read(OUT/'diagnostics/spectral_threshold_counts.json')['groups']
    lines+=['| e_i/e_1 阈值（严格大于） | 全谱方向数：最小/中位/最大 | 90%保护子空间内：最小/中位/最大 |','|---:|---|---|']
    for threshold in ('0.1','0.01','0.001'):
        cells=[]
        for typ in ('full','protected'):
            a=[g['counts'][threshold][typ] for g in counts.values()];cells.append(f'{min(a)}/{np.median(a):g}/{max(a)}')
        lines.append(f'| {threshold} | {cells[0]} | {cells[1]} |')
    lines+=['','## B1 强度网格、命中与算子选择','',
            '| 配置 | 遗忘合并 ROUGE | 遗忘图像 ROUGE | 遗忘文本 ROUGE | 保留图像 ROUGE | 保留文本 ROUGE |','|---|---:|---:|---:|---:|---:|']
    for name in sorted(audit_result['candidates']):
        s=summary(name);lines.append(f'| {name} | {s["forget_combined"]:.6f} | {s["forget_image"]:.6f} | {s["forget_text"]:.6f} | {s["retain_image"]:.6f} | {s["retain_text"]:.6f} |')
    lines+=['','| 族 | 实测状态 | 最终配置 | 遗忘合并 ROUGE | 文本监控差（百分点） | 实测保留图像 ROUGE | 固定目标处插值保留 ROUGE |','|---|---|---|---:|---:|---:|---:|']
    for family in ('C0','C1','C2','C3'):
        m=read(OUT/'matching'/f'{family}.json');s=m.get('summary',{})
        lines.append(f'| {family} | {m["state"]} | {m.get("selected_config",{}).get("name","—")} | {fmt(s.get("forget_combined"))} | {fmt(m.get("text_monitor_difference_pp"),3)} | {fmt(s.get("retain_image"))} | {fmt(m.get("interpolated_retain_at_exact_target"))} |')
    lines+=['',f'算子选择状态：`{operator["state"]}`；选中 `{operator["selected_family"]}`；软减硬的保留图像ROUGE差为 {fmt(operator["soft_minus_hard_retain_image"])}。',
            '只在实际命中且通过文本监控的算子之间，按§6.2比较固定目标处的插值保留指标；差的绝对值<0.01时选软收缩。若硬投影到α=8仍无可行匹配，结论仅限τ=0.9和实测强度范围。',
            f'两个实际命中点的保留差另报为 {fmt(operator["actual_soft_minus_hard_retain_image"])}，不替代同目标比较。最初代码曾按实际点直接比较，已在任何新C2/C3生成前登记实现修正，见operator_comparison_correction.json及OPERATOR_COMPARISON.sha256；原冻结文件未覆盖。',
            '**这是开发集上的ROUGE选择，不构成测试集非劣效或事实性优越的结论。**','',
            '## 耗时与验收','',
            '| 配置 | GPU | 构造秒 | 本次生成墙钟秒 | 逐题生成累计秒 |','|---|---:|---:|---:|---:|']
    for name,c in audit_result['candidates'].items():
        lines.append(f'| {name} | {c["gpu"]} | {c["construction_seconds"]:.1f} | {c["generation_wall_seconds"]:.1f} | {c["generation_sum_question_seconds"]:.1f} |')
    lines+=['','如有恢复，“本次墙钟”只覆盖最后一次运行；逐题累计秒包含保留缓存中每道题的实际生成时间。','',
            'GPU 1 在诊断采集期间提前并发运行一个评测进程；显存预算经试跑核对，推理与构造设置不变。重叠起止时间见 `gpu1_concurrency.json`，墙钟耗时包含资源共享的影响。','',
            f'- 新候选 {audit_result["n_new_candidates"]} 个，合计 {audit_result["n_candidate_questions"]} 道开发题。',
            '- 冻结时间、代码/配置身份、逐题人物与题目、开发边界、B1前λ锁定及最终命中均经程序审计。',
            '- 所有原始输出和大统计留在本地，本报告和审计仅汇总实验结果。',
            '- 详细文件索引见 README.md；结果哈希清单为 RESULTS_supplement.sha256。','']
    lines+=['### 诊断采集耗时与资源','',
            '| 阶段 | 提示条数 | 层数 | 采集秒（含检查点） | 峰值分配显存 GiB | 前缀最大相对变化 | 视觉重复探针最大相对变化 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('pilot','D_fm','D_rh','D_fi'):
        s=read(OUT/'diagnostics'/name/'summary.json')
        lines.append(f'| {name} | {s["samples"]} | {len(s["layers"])} | {s["seconds"]:.1f} | {s["peak_gpu_bytes"]/2**30:.2f} | {s["prefix_relative_max"]:.6f} | {s["visual_duplicate_relative_max"]:.6f} |')
    lines+=['','显存为诊断进程的峰值分配量；不包括并行评测进程。采集计时含定期检查点，最终逐组矩阵落盘可能略晚。','']
    stability=OUT/'stability/results.json'
    if stability.exists():
        rows=read(stability)['groups']
        lines+=['### B1 文本分半稳定性（可选诊断）','',
                '固定种子20261023，按样本类型将128条B1校准样本分成64+64条；沿用含参考答案的原校准输入。两半分别构造文本及一次公共前缀的二阶矩，比较其90%能量子空间。','',
                '| 指标 | 84组最小值 | 中位数 | 最大值 |','|---|---:|---:|---:|']
        for key,label in [('A_basis_coverage_on_B','A子空间覆盖B'),('B_basis_coverage_on_A','B子空间覆盖A'),('overlap_normalized_by_smaller_rank','重叠 / 较小方向数'),('overlap_normalized_by_larger_rank','重叠 / 较大方向数')]:
            a=[r[key] for r in rows];lines.append(f'| {label} | {min(a):.6f} | {np.median(a):.6f} | {max(a):.6f} |')
        lines+=['','这里只评估S1文本分量，样本分半可能共享人物，不是人物留出或总体置信区间，也不替代完整S1稳定性评估。逐组方向数见 `stability/results.json`。','']
    (OUT/'A补点与B1_实验记录及结果.md').write_text('\n'.join(lines))

def main():
    result=audit()
    from wp1_supplement.plot_results import make_plot
    make_plot();report(result)
    include=[OUT/'README.md',OUT/'FREEZE_supplement.sha256',OUT/'completion_audit.json',OUT/'A补点与B1_实验记录及结果.md']
    include+=list(OUT.glob('*.py'))
    for name in ('c1_eight_point_selection.json','c1_selection.json','c1_bootstrap.json','B1_selection_inputs.json','B1_selection_inputs.sha256','B1_operator_selection.json','B1_baselines.json','workflow_unit_checks.json','rule_numeric_interpretation.json'):
        include.append(OUT/name)
    include+=list((OUT/'matching').glob('*.json'))
    include+=list(OUT.glob('gpu1_concurrency.json'))
    include+=list((OUT/'stability').glob('*.json'))+list((OUT/'stability').glob('*.sha256'))
    include+=list((OUT/'stability').glob('half_*/summary.json'))
    include+=list(OUT.glob('诊断与灵敏度检查结果.md'))
    include+=list(OUT.glob('C1补点与重选结果.md'))
    include+=[OUT/'operator_comparison_correction.json',OUT/'OPERATOR_COMPARISON.sha256']
    include+=list((OUT/'implementation_archive_before_6p2_correction').glob('*'))
    include+=list((OUT/'figures').glob('*'))
    include+=list((OUT/'appearance').glob('*/sensitivity.json'))+list((OUT/'appearance').glob('*.json'))
    include+=[OUT/'diagnostics'/name for name in ('unit_checks.json','scope_extension.json','status.json','spectral_threshold_counts.json','coverage.json','response_ratios.json')]
    include+=[OUT/'diagnostics'/name/'summary.json' for name in ('pilot','D_fm','D_rh','D_fi')]
    include+=list((OUT/'diagnostics/pilot').glob('*validation.json'))
    for path in (OUT/'candidates').glob('*/metrics.json'):include+=[path,path.parent/'status.json',path.parent/'manifest.json',path.parent/'construction.json']
    inventory=''.join(sha256(p)+'  '+str(p.relative_to(OUT))+'\n' for p in sorted(set(include)))
    (OUT/'RESULTS_supplement.sha256').write_text(inventory)
    print(json.dumps({k:result[k] for k in ('state','n_new_candidates','matched_families','unmatched_families','selected_operator')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
