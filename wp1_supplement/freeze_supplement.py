"""Freeze input-only supplementary rules before any new model generation."""
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wp1_a_stage.common import DATA, frozen, questions_for_row, rows
from wp1_review.prepare_wp1 import normalize

OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    if (OUT / 'FREEZE_supplement.sha256').exists():
        raise RuntimeError('Supplement already frozen; never overwrite')
    part = frozen()
    old = ROOT / 'wp1_a_stage'
    templates = json.loads((old / 'templates.json').read_text())
    templates['appearance_holdout_backup'] = [
        dict(attribute='facial_detail_focus', question='Are the fine edges of the eyelashes and irises sharply resolved or softly blurred?'),
        dict(attribute='directional_motion_blur', question='Is directional motion blur visibly smearing the facial details?'),
        dict(attribute='chromatic_aberration', question='Are coloured fringes visible along high-contrast facial edges?'),
        dict(attribute='sharpening_halos', question='Are bright or dark digital sharpening halos visible along the outline of the nose?'),
        dict(attribute='interlacing_artifact', question='Are horizontal scan lines or interlacing stripes visible across the facial region?'),
        dict(attribute='privacy_redaction', question='Is the facial region deliberately obscured by a mosaic or a privacy-blur effect?'),
    ]
    templates['publication_history'] = dict(original_templates_sha256=sha(old / 'templates.json'), original_file_unchanged=True,
        backup_policy='One whole-bank replacement iff the prespecified MFA positive-control sensitivity test fails; no question-wise selection.')
    frame_forget = rows(DATA / 'eval_data/forget_5/train-00000-of-00001.parquet')
    frame_retain = rows(DATA / 'eval_data/retain_5/train-00000-of-00001.parquet')
    allq = sorted({q['Question'] for frame in [frame_forget, frame_retain] for _, row in frame.iterrows() for q in row['image understanding']})
    stop = set('a an the is are does do what which how of in on at to or and with this that there any have has person persons photo image picture it its be'.split())
    def tokens(s):
        return set(normalize(s).split()) - stop
    def sim(a, b):
        x, y = tokens(a), tokens(b)
        return len(x & y) / max(1, len(x | y))
    checks = []
    for group in ['B2', 'appearance_holdout', 'appearance_holdout_backup']:
        for item in templates[group]:
            near = sorted([(sim(item['question'], q), q) for q in allq], reverse=True)[:3]
            checks.append(dict(group=group, **item, nearest=[dict(jaccard=s, question=q) for s, q in near],
                exact_overlap=normalize(item['question']) in {normalize(q) for q in allq},
                semantic_review='Pass. B2 asks frame geometry/head roll; primary/backup ask image encoding, overlays or optical/digital degradation. Backup focus is facial resolution, not background focus, eye colour/opening, expression, skin condition or clothing. No identity/age/gender/ethnicity.'))
    assert all(not c['exact_overlap'] and c['nearest'][0]['jaccard'] < .5 for c in checks)
    cross = [dict(b2=a['attribute'], holdout=b['attribute'], jaccard=sim(a['question'], b['question']),
                  semantic_review='Different task: crop/position/roll versus rendering/quality; no paraphrase.')
             for a in templates['B2'] for group in ['appearance_holdout', 'appearance_holdout_backup'] for b in templates[group]]
    save(OUT / 'templates.json', templates)
    save(OUT / 'template_similarity.json', dict(method='Same content-word Jaccard as original freeze; reject >=0.5 and exact matches; semantic task review',
        checks=checks, cross_b2_holdout=cross, semantic_overlap=False, eval_unique_questions=len(allq)))
    rng = random.Random(20261023)
    selected = {}
    for label, ids, frame, kinds in [
        ('D_fm', part['forget']['F1_development'], frame_forget, ['image_memory', 'text_memory']),
        ('D_fi', part['forget']['F1_development'], frame_forget, ['image_understanding']),
    ]:
        records = []
        for i, row in frame[frame.ID.isin(ids)].iterrows():
            records += [q for q in questions_for_row(row, 'forget', i) if q['question type'] in kinds and (label != 'D_fi' or q['image_type'] == 'character')]
        selected[label] = records
    # H never contributes to calibration or selection. Balanced prompt types, 150 people.
    h_ids = sorted(rng.sample(part['retain']['H_pool'], 150))
    records = []
    for i, row in frame_retain[frame_retain.ID.isin(h_ids)].iterrows():
        qs = questions_for_row(row, 'retain', i)
        for kind in ['image_memory', 'text_memory']:
            records.append(rng.choice([q for q in qs if q['question type'] == kind]))
    selected['D_rh'] = records
    assert {k: len(v) for k, v in selected.items()} == dict(D_fm=325, D_fi=125, D_rh=300)
    save(OUT / 'diagnostic_selection.json', dict(seed=20261023, layers=list(range(0, 28, 3)), selected=selected,
        source_sha256={str(p.relative_to(DATA)): sha(p) for p in [DATA / 'eval_data/forget_5/train-00000-of-00001.parquet', DATA / 'eval_data/retain_5/train-00000-of-00001.parquet']}))
    now = datetime.now(timezone.utc).isoformat()
    protocol = dict(frozen_at=now, parent_freeze_sha256=sha(old/'FREEZE.sha256'), parent_results_sha256=sha(old/'RESULTS.sha256'),
        target=0.4482903966036022, hit_range=[0.44029, 0.45229], margin_pp=2.0, delta_pp=5,
        channel_upper_limits_pp=dict(image_memory=10, text_memory=10), text_monitor_pp=5, soft_tie_rouge=.01,
        initial_c1=[.8,.85,.9,.95], lambda_rule='Original v3 6.1 on original four plus new four C1 points; then recompute with all frozen adaptive target-bracketing points. Keep both eight-point and all-point selections. Ties 1e-8 smaller lambda.',
        bracket_rule='At target, nearest observed x below/above. If x gap >0.1, midpoint the corresponding strength parameters, at most two additions per family. No extrapolation. Same x choose higher retain image ROUGE; ties smaller parameter.',
        soft_alphas=[1.25,1.5,2,3], hard_alphas=[1.5,2,3], hard_extensions=[5,8],
        hard_failure='No feasible hard projection at tau=0.9 within measured strength range/budget; not a universal failure claim.',
        final_rule='Interpolate strength at target from bracketing observed x; run that configuration on all 2375 development questions. Require hit range and text residual <= SMFA+5pp. At most two additional secant refinements per family. If text monitor fails, try stronger target 0.44129 inside fixed hit range. Never rank unvalidated/unmatched configurations as winners. No baseline retention used to tune target.',
        baseline_validation='C0 k from old 1/5/10 curve interpolation (about 5.3); k_text=2*k_multi; C1 from enriched curve. Each gets an actual target-strength development evaluation; same matching/refinement/text checks.',
        operator_rule='Both eligible: abs(retain_image soft-hard)<0.01 selects soft, else higher actual retain image. Only one eligible selects it. Neither eligible: unmatched/no selection. No factuality claim.',
        bootstrap=dict(replicates=20000,seed=20261023,unit='person',forget_retain='independent resamples; same model-cohort draws',interpolation='recompute each draw; no extrapolation',limitation='Conditional on measured knots and selected candidates; excludes curvature and selection-process error.'),
        sensitivity=dict(models=['Original','SMFA','MFA_unsculpted'],people='F1 only,25',templates_per_person=6,
                         positive_control='MFA_unsculpted versus Original; SMFA separately reported',
                         insensitive_rule='Loose refusal increment <0.02 AND mean ROUGE-L agreement with Original >=0.90 on all 150 prompts. Exact equality at 0.02 is sensitive; ROUGE=0.90 qualifies. Also report Original-refusal/nonrefusal subsets and per-person intervals.',
                         replacement='If primary insensitive, use entire pre-frozen backup bank once, rerun all three baselines, retain old evidence, record protocol revision. No further template changes if backup also insensitive.'),
        diagnostic='Only Original forward moments. Ten fixed layers (0,3,...27), qkv/o/mlp. Visual unique, text_all, text_image, text_pure separately; common prefix excluded from diagnostics. Compute S0/S0d/S1 coverage and current C2/C3 response ratios. No use in parameter/template selection except prespecified baseline sensitivity rule.',
        folds='Second fold uses same m=2.0 and supplement rules, independent selection; not started in this request. Freeze first-fold choices before fold2; do not alter rules from first confirmation results.',
        scope='Only F1 + retain-development candidates; H-pool diagnostic only; no F2/test-retain candidate generation, no API scoring; stop after B1 and report. Optional B1 half-stability deferred unless resources allow.')
    save(OUT / 'protocol.json', protocol)
    (OUT / '补充冻结记录.md').write_text(f'''# A 补点与 B1 补充冻结记录

冻结时间（UTC）：{now}

本记录先于任何本补充阶段的新模型回答生成。原 A 阶段文件完全保留；本目录 templates.json 是新增备用字段的扩展副本。

- 最终目标 0.4482903966；允许实测命中范围 [0.44029, 0.45229]，向更强遗忘一侧留更宽容差。余量 m=2.0、δ=5、单通道10、文本监控+5、软硬差0.01不变。
- C1 先补0.8/0.85/0.9/0.95；保留原四点、八点、补夹点后的全部曲线选择记录。λ按原6.1重选。
- 目标最近夹点的遗忘ROUGE间距>0.1则二分强度，最多两次，不外推。最终α/c/k需全开发集实测；最多两次命中修正，仍不满足就记匹配不足。
- C2补α=1.25/1.5/2/3；C3补1.5/2/3，必要时5/8；C0及C1目标各实测一次，必要的有限修正同上。
- 灵敏度正对照为未雕刻MFA相对Original：宽松拒答增幅<2百分点且全150题平均一致度ROUGE≥0.90判不灵敏。SMFA并报；Original拒答题另列。触发后只允许一次整套切换，备用6题已预定；不按题目挑选。
- D_fm325、D_fi125、D_rh300题，固定层0/3/.../27；诊断分视觉、带图文本和纯文本，结果不用于选参。S0d/S1由已有B1统计构造。
- 人物bootstrap20,000次、种子20261023；每次重插值，不含曲率误差或选择流程不确定性。
- 第二折沿用m=2.0及同一选择规则；本次不启动第二折、不打分、不进入B2。

精确边界、有限补点预算、备用模板与诊断样本见protocol.json、templates.json、diagnostic_selection.json。数据集题目、模型和大统计留在本地，不上传Git。
''')
    paths = [OUT / name for name in ['freeze_supplement.py','protocol.json','templates.json','template_similarity.json','diagnostic_selection.json','补充冻结记录.md']]
    (OUT / 'FREEZE_supplement.sha256').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in paths))
    print(json.dumps(dict(frozen_at=now, freeze_sha256=sha(OUT/'FREEZE_supplement.sha256'), max_jaccard=max(c['nearest'][0]['jaccard'] for c in checks)),indent=2))


if __name__ == '__main__':
    main()
