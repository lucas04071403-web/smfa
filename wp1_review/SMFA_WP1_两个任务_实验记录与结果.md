# SMFA 工作点 1：第 0 步补齐与 CPU 准备——实验记录及结果

报告生成时间：2026-10-23T08:43:16.377919+00:00。所有执行时间均为 UTC。工作目录：`/root/lucas4`。

## 1. 完成范围与结论

本报告对应两个任务：（1）在两张 GPU 上并行补齐 Base 和未雕刻 MFA 的完整评测、同模型评分及耗时；（2）冻结人物划分、将图像理解拆为人脸与眼科并计算拒答率、实现 LoRA 读取和内存构造并完成自检。两项任务已完成。本文不是 WP1 全部候选方法的最终实验，不包含尚未执行的候选搜索或多随机种子显著性检验。

每个新增模型均完成 19,000 道生成题、19,000 条事实性评分、19,000 条有意义性评分，无缺失分数。评分按索引、生成摘要及最终导出文件进行完整性核验。Original 与 SMFA 使用已有第 0 步结果，本次未重新生成或重新评分。

**耗时更正：**上一条完成通知中的 MFA 评分“12小时3分27秒”错误地重复累计了恢复尝试 2–6；按同一阶段、日志路径和起始时间去重后，正确累计执行耗时为 **7小时53分30秒（28,410.348秒）**。模型输出与评分均未改变。原始状态和耗时已备份，修正证据见 [timing_dedup_audit.json](step0_baselines/timing_dedup_audit.json)。

## 2. 环境、数据与模型

| 项目 | 设置 |
|---|---|
| GPU | 2 × NVIDIA A40，单卡约 46 GB；GPU 0：Base，GPU 1：MFA |
| 软件 | PyTorch 2.8.0+cu128；Python 环境 /root/anaconda3/envs/smllmun |
| 数据 | S-MLLMUn/data/S-MLLMUn-data；forget_ratio=5 |
| 生成 | 官方评测逻辑；FP16；do_sample=False；max_new_tokens=100 |
| 评分 | qwen-plus，DashScope 接口；官方事实性与有意义性提示词，0–10 分；每模型 4 workers |
| 精度与调用说明 | API 服务使用 qwen-plus 名称，未锁定服务端模型快照；历史与本次评分存在时间差 |

| 模型/输入 | 相对 /root/lucas4 的路径 |
|---|---|
| Base | S-MLLMUn/model/base_model/llava-onevision-qwen2-7b-si-hf |
| Original | S-MLLMUn/model/original_model/llava-onevision-qwen2-7b-original |
| 保存的 SMFA | S-MLLMUn/model/unlearned_model/llava-onevision/split_ratio_5/SMFA/sculpted |
| 本次使用适配器 | S-MLLMUn/model/unlearned_model/llava-onevision/split_ratio_5/SMFA/epoch_3/{MFA_multi,MFA_text,RA} |
| 未雕刻 MFA | S-MLLMUn/model/unlearned_model/llava-onevision/split_ratio_5/MFA_unsculpted/sculpted |

### 2.1 评测规模

| 集合 | 人物数 | 图像理解 | 图像记忆 | 文本记忆 | 合计 |
|---|---|---|---|---|---|
| forget | 50 | 300（人脸250、眼科50） | 350 | 300 | 950 |
| retain | 950 | 5700（人脸4750、眼科950） | 6650 | 5700 | 18050 |

完整第 0 步评测覆盖全部 retain 人物；WP1 后续开发/测试划分与这次完整基线评测范围不同，不能把全量 retain 结果当作独立测试集调参结果。

## 3. 任务一：并行评测执行记录

### 3.1 未雕刻 MFA 构造

直接运行仓库官方 `unlearning_methods/SMFA_sculpt.py`，适配器使用 epoch_3，设置 `--multi_k 1e20 --text_k 1e20`。没有把 MFA 替换成单个适配器或训练中间模型。官方脚本日志确认：

```text
[multi] masked parameters: 0/6628802560, ratio: 0.0000
[text]  masked parameters: 0/6628802560, ratio: 0.0000
```

因此该配置在本次权重上确实没有雕刻。保存模型包含 4 个分片、765 个张量。证据：[零掩码核验](step0_baselines/MFA_unsculpted_zero_mask.json)。

### 3.2 时间线与耗时

| 模型 | 阶段 | 尝试 | 开始 UTC | 结束 UTC | 秒 | 状态 |
|---|---|---|---|---|---|---|
| Base | evaluate | 1 | 2026-10-22 06:29:37 | 2026-10-22 18:03:21 | 41623.955 | complete |
| Base | score | 1 | 2026-10-22 18:03:21 | 2026-10-23 02:17:17 | 29635.724 | complete |
| MFA_unsculpted | sculpt | 1 | 2026-10-22 06:22:59 | 2026-10-22 06:50:22 | 1643.204 | complete |
| MFA_unsculpted | evaluate | 1 | 2026-10-22 06:50:24 | 2026-10-22 14:53:19 | 28974.072 | complete |
| MFA_unsculpted | score | 1 | 2026-10-22 14:53:19 | 2026-10-22 18:36:44 | 13405.616 | failed |
| MFA_unsculpted | score | 2 | 2026-10-23 00:41:24 | 2026-10-23 00:41:34 | 9.964 | failed |
| MFA_unsculpted | score | 3 | 2026-10-23 00:42:04 | 2026-10-23 04:51:04 | 14939.115 | failed |
| MFA_unsculpted | score | 4 | 2026-10-23 04:51:34 | 2026-10-23 04:51:49 | 15.506 | failed |
| MFA_unsculpted | score | 5 | 2026-10-23 04:52:19 | 2026-10-23 04:52:35 | 15.776 | failed |
| MFA_unsculpted | score | 6 | 2026-10-23 04:53:05 | 2026-10-23 04:53:22 | 16.571 | failed |
| MFA_unsculpted | score | 7 | 2026-10-23 08:19:28 | 2026-10-23 08:19:36 | 7.800 | complete |

| 模型 | 完整评测 | 评分累计执行时间 | 雕刻（单独计时） |
|---|---|---|---|
| Base | 11小时33分44秒 | 8小时13分56秒 | 不适用 |
| MFA_unsculpted | 8小时2分54秒 | 7小时53分30秒 | 0小时27分23秒 |

计时口径：评测包含加载、生成和 ROUGE 汇总；评分包含 API 请求、等待及子进程内重试。失败尝试计入评分累计耗时，但两次启动之间无人执行的空闲时间、等待人工恢复的时间不计入。雕刻与 SMFA 答案自检不混入评测耗时。MFA 从首次评分到最终补齐跨越约17小时26分，但累计运行时间为7小时53分30秒；后续排期应同时预留 API 重试及故障恢复余量。完整秒数见 [timings.json](step0_baselines/timings.json)。

### 3.3 异常处理及实验影响

- 初始会话看不到 GPU；在用户开启完全访问后恢复。按用户明确要求清理原 GPU 占用后再启动本任务。
- 官方雕刻持有多份稠密更新，内存需求超过本机约128 GiB，因此临时添加64 GiB swap并短暂停止/恢复雕刻进程；未修改公式或官方脚本。swap于2026-10-22 08:13:33 UTC清理完毕。雕刻耗时包含此影响。
- 评分从现有 Conda 激活脚本加载凭据；只在任务进程清除不兼容的代理环境变量，密钥未写入日志。
- MFA 首次事实性评分缺3条；恢复后全部补齐。最终有意义性评分缺2条（retain索引9404、14268），多轮重试后仍因响应中的非法JSON单引号转义被官方解析器拒收。
- 最终只在独立恢复入口兼容非法 `\'` 转义，保留同一模型、输入、提示词和分值验证规则，重新请求并补齐两条。没有从日志挑选分数或人工指定分数；已有效的缓存没有重打。该两条解析路径与其余题目不同，作为可复现性限制明确记录。
- 所有失败日志与恢复记录均保留；最终按每题索引和内容摘要核对，导出结果与缓存一致。

## 4. 任务一：完整第 0 步结果

ROUGE-L 为0–1，事实性/有意义性为0–10。下表来自各模型完整导出结果，未用部分评分估算。图像理解的人脸/眼科细分见第6节。

| 模型 | 集合 | 任务 | 题数 | ROUGE-L | 事实性 | 有意义性 |
|---|---|---|---|---|---|---|
| Original | forget | 图像理解 | 300 | 0.6823 | 7.230 | 9.297 |
| Original | forget | 图像记忆 | 350 | 0.6820 | 7.477 | 9.223 |
| Original | forget | 文本记忆 | 300 | 0.7405 | 8.500 | 8.460 |
| Original | retain | 图像理解 | 5700 | 0.6937 | 7.353 | 9.374 |
| Original | retain | 图像记忆 | 6650 | 0.7052 | 7.499 | 9.222 |
| Original | retain | 文本记忆 | 5700 | 0.7623 | 8.777 | 8.508 |
| SMFA | forget | 图像理解 | 300 | 0.6274 | 6.463 | 9.357 |
| SMFA | forget | 图像记忆 | 350 | 0.3631 | 3.643 | 9.397 |
| SMFA | forget | 文本记忆 | 300 | 0.4919 | 5.650 | 8.993 |
| SMFA | retain | 图像理解 | 5700 | 0.6595 | 6.889 | 9.401 |
| SMFA | retain | 图像记忆 | 6650 | 0.5802 | 5.980 | 9.255 |
| SMFA | retain | 文本记忆 | 5700 | 0.7180 | 8.264 | 8.535 |
| Base | forget | 图像理解 | 300 | 0.3911 | 7.123 | 9.580 |
| Base | forget | 图像记忆 | 350 | 0.2394 | 0.529 | 9.323 |
| Base | forget | 文本记忆 | 300 | 0.3931 | 0.230 | 8.923 |
| Base | retain | 图像理解 | 5700 | 0.4064 | 7.430 | 9.558 |
| Base | retain | 图像记忆 | 6650 | 0.2455 | 0.493 | 9.306 |
| Base | retain | 文本记忆 | 5700 | 0.4009 | 0.364 | 8.857 |
| MFA_unsculpted | forget | 图像理解 | 300 | 0.5606 | 5.690 | 9.353 |
| MFA_unsculpted | forget | 图像记忆 | 350 | 0.1498 | 1.346 | 9.186 |
| MFA_unsculpted | forget | 文本记忆 | 300 | 0.2273 | 2.457 | 9.113 |
| MFA_unsculpted | retain | 图像理解 | 5700 | 0.6239 | 6.422 | 9.360 |
| MFA_unsculpted | retain | 图像记忆 | 6650 | 0.3643 | 3.617 | 9.189 |
| MFA_unsculpted | retain | 文本记忆 | 5700 | 0.6569 | 7.532 | 8.590 |

## 5. 任务二：固定人物划分（§2.2）

使用人物ID排序、`random.Random(42)`洗牌和固定顺序切片，未依据模型得分调整人物归属；划分已冻结，此后没有重抽。输入parquet的SHA-256和完整名单已存档。

| 集合 | 划分 | 人物数 |
|---|---|---|
| forget | F1_development | 25 |
| forget | F2_test | 25 |
| retain | B1_few_shot_excluded | 50 |
| retain | development | 100 |
| retain | test | 200 |
| retain | X_pool | 300 |
| retain | H_pool | 300 |

B1的50人为RA实际少样本训练人物，从开发、测试、X/H池中排除。各分组互斥，forget共50人，retain共950人，全覆盖1000人。

**协议偏差：**冻结划分前，为确认JSON结构曾查看少量SMFA生成记录，因此不能宣称严格满足“在看任何逐人结果之前冻结”的完全前瞻盲化要求。划分只依赖parquet ID与固定种子，未用分数筛人；该情况已写入manifest，未事后重抽。本报告也包含已有全量基线结果，后续不应依据F2/test逐人表现选择参数。

划分文件SHA-256：`ff355f2030c641a03d9320a867eff2f6c383ef6dcf197203ac6e66eff0ecbb46`。

证据：[冻结名单](fixed_person_split.json)、[划分校验](fixed_person_split_validation.json)。

## 6. 任务二：图像理解拆分与拒答率（§2.3）

按源parquet及image_type分为face和ophthalmology，逐题附加person_id、cohort、generation_index。评分优先按JSONL索引、生成内容摘要及完整问题元组关联，不依赖多线程评分输出顺序。下表仅覆盖image_understanding；不把图像记忆或文本记忆混入。

拒答定义：精确拒答对大小写和标点归一化后匹配IDK.py/IDK_extend.py短语；宽松拒答额外匹配显式拒答正则。它们是词法规则指标，不等同于人工语义标注；未匹配不能证明模型没有拒答。

| 模型 | 集合 | 图像类型 | 题数 | 事实性 | 有意义性 | ROUGE-L | 精确拒答 | 宽松拒答 | 平均词数 |
|---|---|---|---|---|---|---|---|---|---|
| Original | forget | face | 250 | 6.892 | 9.756 | 0.6387 | 0 (0.00%) | 0 (0.00%) | 8.23 |
| Original | forget | ophthalmology | 50 | 8.920 | 7.000 | 0.8999 | 0 (0.00%) | 0 (0.00%) | 10.04 |
| Original | retain | face | 4750 | 7.005 | 9.836 | 0.6492 | 0 (0.00%) | 0 (0.00%) | 8.69 |
| Original | retain | ophthalmology | 950 | 9.094 | 7.063 | 0.9166 | 0 (0.00%) | 0 (0.00%) | 10.26 |
| SMFA | forget | face | 250 | 6.380 | 9.748 | 0.6122 | 5 (2.00%) | 5 (2.00%) | 8.23 |
| SMFA | forget | ophthalmology | 50 | 6.880 | 7.400 | 0.7032 | 9 (18.00%) | 10 (20.00%) | 9.02 |
| SMFA | retain | face | 4750 | 6.654 | 9.845 | 0.6279 | 35 (0.74%) | 35 (0.74%) | 8.35 |
| SMFA | retain | ophthalmology | 950 | 8.066 | 7.180 | 0.8172 | 70 (7.37%) | 79 (8.32%) | 10.14 |
| Base | forget | face | 250 | 8.364 | 9.948 | 0.4634 | 0 (0.00%) | 0 (0.00%) | 7.84 |
| Base | forget | ophthalmology | 50 | 0.920 | 7.740 | 0.0294 | 0 (0.00%) | 0 (0.00%) | 82.48 |
| Base | retain | face | 4750 | 8.705 | 9.941 | 0.4797 | 0 (0.00%) | 0 (0.00%) | 8.41 |
| Base | retain | ophthalmology | 950 | 1.054 | 7.643 | 0.0397 | 0 (0.00%) | 0 (0.00%) | 81.64 |
| MFA_unsculpted | forget | face | 250 | 5.684 | 9.680 | 0.5557 | 16 (6.40%) | 16 (6.40%) | 8.61 |
| MFA_unsculpted | forget | ophthalmology | 50 | 5.720 | 7.720 | 0.5848 | 17 (34.00%) | 18 (36.00%) | 9.06 |
| MFA_unsculpted | retain | face | 4750 | 6.251 | 9.784 | 0.6005 | 125 (2.63%) | 126 (2.65%) | 8.42 |
| MFA_unsculpted | retain | ophthalmology | 950 | 7.282 | 7.240 | 0.7409 | 144 (15.16%) | 148 (15.58%) | 9.27 |

逐题结果存于 `{Original,SMFA,Base,MFA_unsculpted}_{forget,retain}_image_understanding_split.json`；按冻结cohort的统计也包含在文件中。汇总见 [existing_i_u_split_summary.json](existing_i_u_split_summary.json)。

## 7. 任务二：LoRA读取、内存构造及自检（§2.5）

实现文件：[memory_construct.py](memory_construct.py)。读取epoch_3的MFA_multi、MFA_text、RA共274个适配层，按LoRA的alpha/r缩放因子计算B×A。RA仅用于雕刻掩码判断；最终更新为雕刻后的multi/text两项平均，再加到Original权重。每次候选构造重新读取Original适配层，不在上一候选的FP16权重上回减，避免累计误差；支持逐tensor生成与直接写入内存模型，不必先保存完整候选checkpoint。

计算约定：`Δ = (alpha/r) · B @ A`；逐层 `ratio = ||ΔF|| / (||ΔR|| + 1e-12)`；当`ΔR·ΔF < 0`且`k·ratio·|ΔR| < |ΔF|`时将对应ΔF置零。最终`W = FP16(FP16(W_original) + (sculpt(Δmulti)+sculpt(Δtext))/2)`。SMFA自检使用k_multi=5、k_text=10。

精度按本机已验证的PEFT加载行为实现：第一个MFA_multi适配器因子保留FP32，后加载MFA_text/RA因子先转FP16再回FP32；更新及掩码在FP32计算，合并后转FP16。Original未适配张量按官方FP16加载后的值比较，因此FP32保存的image_newline需先转FP16。修正前曾因未转换该张量出现自检失败，已按官方加载语义修正，源码快照及AST审计保留；权重构造和答案生成函数未变。

| 检查项 | 结果 |
|---|---|
| 适配层逐元素匹配保存SMFA | 274/274，通过 |
| 未适配张量匹配Original的FP16加载值 | 491/491，通过 |
| SMFA multi掩码参数数 | 423,559,537 |
| SMFA text掩码参数数 | 212,599,596 |
| 现有SMFA题目覆盖 | forget 950；retain 18050 |
| F1前30题答案完全一致 | 30/30，一致率100%，超过27/30门槛 |
| CPU权重自检耗时 | 473.487秒 |
| 答案比对内部耗时 | 374.466秒（含子进程启动整体398.285秒） |

30题在冻结F1开发人物中按源数据顺序选取，selection单独存档；比较内存构造模型的生成字符串与第0步保存SMFA结果。该自检只使用Original、适配器、SMFA checkpoint及其既有评测结果，不等待Base/MFA完成。

证据：[权重自检](memory_construct_self_check.json)、[30题选择](memory_construct_answer_selection.json)、[答案比对](memory_construct_answer_check.json)、[精度修正审计](memory_construct_precision_correction.json)、[准备代码验证](preparation_validation.json)。

## 8. 结果解读与下一步依据

1. SMFA相对未雕刻MFA降低了保留集眼科宽松拒答率（15.58%→8.32%），眼科事实性提高（7.282→8.066）；但仍未恢复到Original的9.094。人脸保留集也有相同方向改善（宽松拒答2.65%→0.74%，事实性6.251→6.654）。
2. 遗忘集眼科宽松拒答率由MFA的36%降至SMFA的20%，事实性由5.720提高至6.880。图像理解恢复与记忆遗忘是不同指标，不能把这里事实性回升直接判定为“遗忘失败”，应结合图像/文本记忆结果。
3. 未雕刻MFA的遗忘集图像记忆ROUGE-L为0.1498，SMFA为0.3631；保留集分别为0.3643与0.5802，体现雕刻后的保留能力恢复及遗忘/保留权衡。ROUGE下降本身不充分证明有效遗忘，需联合事实性与拒答表现。
4. Base眼科事实性很低（forget0.920、retain1.054）且回答较长（约82词），尽管词法拒答率为0；因此拒答率不能单独代表任务能力，有意义性也不能替代事实性。
5. 后续WP1应同时监控人脸与眼科，不只看图像理解总均值。当前结果是该次checkpoint和数据划分上的描述性比较，尚无多种子/统计显著性证据，不能宣称WP1方法已得到最终验证。

## 9. 复核入口与交付文件

```bash
# 已完成实验的CPU重新汇总（不调用评分API）
/root/anaconda3/envs/smllmun/bin/python /root/lucas4/wp1_review/prepare_wp1.py --analyze-existing
# 准备代码验证
/root/anaconda3/envs/smllmun/bin/python /root/lucas4/wp1_review/validate_preparation.py
```

GPU生成结果与评分JSON：`S-MLLMUn/eval_result/llava-onevision/split_ratio_5/{Base,MFA_unsculpted}/`；Original/SMFA同级目录。每模型含forget/retain生成文件、两项逐题JSONL缓存、两项最终score.json与ROUGE汇总。

| 交付物 | 路径 |
|---|---|
| 最终任务状态 | [status.json](step0_baselines/status.json) |
| 更正后的最终耗时 | [timings.json](step0_baselines/timings.json) |
| 官方零掩码证据 | [MFA_unsculpted_zero_mask.json](step0_baselines/MFA_unsculpted_zero_mask.json) |
| 最终两条评分恢复与核验 | [final_score_recovery.json](step0_baselines/final_score_recovery.json) |
| 只修复非法JSON转义的入口 | [score_json_recovery_runner.py](score_json_recovery_runner.py) |
| 冻结人物划分 | [fixed_person_split.json](fixed_person_split.json) |
| 四模型图像理解报告 | [step0_image_understanding_report.md](step0_image_understanding_report.md) |
| LoRA内存构造代码 | [memory_construct.py](memory_construct.py) |

历史`scoring_recovery.json`保留第一次恢复耗尽重试时的failed状态，属于历史证据；最终状态以`status.json`与`final_score_recovery.json`为准。所有计时修正只改状态元数据，不改变原始日志、模型、逐题输出或分数。
