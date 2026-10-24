# SMFA 复现与工作点 1 实验

本仓库保存 SMFA 复现代码、WP1 实验设计、实验记录和小型结果摘要。数据集、模型权重、训练适配器、逐题输出、评分缓存及运行日志由实验主机本地保存，不包含在 Git 历史中。

## 先看这些文档

| 内容 | 文档 |
|---|---|
| A 阶段实验记录及结果 | [SMFA_WP1_A阶段_实验记录与结果.md](wp1_a_stage/SMFA_WP1_A阶段_实验记录与结果.md) |
| WP1 实验设计 v3 | [SMFA_WP1_experiment_design_v3.md](SMFA_WP1_experiment_design_v3.md) |
| A 阶段冻结规则 | [开工前冻结记录.md](wp1_a_stage/开工前冻结记录.md) |
| 版本恢复方法 | [ROLLBACK.md](ROLLBACK.md) |
| 本次两个任务的完整记录及结果 | [SMFA_WP1_两个任务_实验记录与结果.md](wp1_review/SMFA_WP1_两个任务_实验记录与结果.md) |
| 后续建议审核与 CPU bootstrap 结果 | [SMFA_WP1_Claude建议审核.md](wp1_review/SMFA_WP1_Claude建议审核.md) |
| WP1 实验设计 v2 | [SMFA_WP1_experiment_design_v2.md](SMFA_WP1_experiment_design_v2.md) |
| 三个工作点 | [SMFA_three_work_points_v2_1.md](SMFA_three_work_points_v2_1.md) |
| 初始 SMFA 复现结果 | [SMFA_复现实验结果.md](SMFA_复现实验结果.md) |
| 四模型人脸/眼科理解汇总 | [step0_image_understanding_report.md](wp1_review/step0_image_understanding_report.md) |
| WP1 准备代码与使用说明 | [wp1_review/README.md](wp1_review/README.md) |

## 已完成实验

Base 和未雕刻 MFA 均完成 19,000 题生成及两项各 19,000 条的 qwen-plus 评分。未雕刻 MFA 使用官方雕刻脚本，两个 `k` 均为 `1e20`，日志核验两项掩码数均为 0。

| 模型 | 完整评测耗时 | 评分累计执行耗时 |
|---|---:|---:|
| Base | 11 小时 33 分 44 秒 | 8 小时 13 分 56 秒 |
| MFA（不雕刻） | 8 小时 2 分 54 秒 | 7 小时 53 分 30 秒 |

评分耗时累计实际运行及重试，不含失败后等待恢复的空闲时间。MFA 曾出现恢复记录重复累计，已更正；以 [timings.json](wp1_review/step0_baselines/timings.json) 和 [修正审计](wp1_review/step0_baselines/timing_dedup_audit.json) 为准。

人物划分保持固定。LoRA 内存构造验证了 274 个适配层及 491 个未适配张量，30 道题的生成答案全部匹配。详见实验报告中对精度、划分协议偏差、评分 JSON 解析修复及统计结论范围的说明。

A 阶段完成 10 个候选，每个仅在开发集评测 2,375 题，共 23,750 条回答；另完成 975 条同期事实性评分。候选未评测 F2 或测试保留集。按冻结规则选择 λ=10，规划区间半宽 h=4.302 个百分点，余量为 2 个百分点，下一阶段遗忘合并 ROUGE 目标为 0.448290。当前 C2 的 α=1 在事实性残余上较 SMFA 恶化 13.472 个百分点，不能判定工作点成功；详见报告中的约束和统计局限。本次归档不启动 B 阶段。

## 代码布局

```text
S-MLLMUn/       上游项目的源文件快照（无嵌套 .git、数据集或权重）
reproduce_smfa/ 训练、雕刻、评测与可恢复评分入口
smfa_run/       复现核查、运行辅助代码与选定摘要
wp1_review/     固定划分、指标拆分、内存构造、评分恢复、统计审核
wp1_a_stage/    A 阶段冻结规则、候选构造、统计算子与选定结果汇总
scripts/        提交内容检查
```

上游来源与版本见 [THIRD_PARTY.md](THIRD_PARTY.md)。[PUBLISH_MANIFEST.json](PUBLISH_MANIFEST.json) 记录发布源文件的哈希、大小和内容裁剪；未为本仓库新增对上游代码的许可证声明。

## 本地数据与环境

原实验在 `/root/lucas4`、Conda 环境 `/root/anaconda3/envs/smllmun`、两张 A40 上执行。`smfa_environment.yml` 保存依赖版本；发布时移除了该环境文件的主机专用 `prefix`。部分脚本仍记录原主机绝对路径，迁移时请先核对路径和环境，不要直接复用历史状态文件恢复新实验。

请另行下载并放置以下资产；本仓库没有下载它们的副本：

| 资产 | 放置位置（相对项目根目录） |
|---|---|
| [S-MLLMUn 数据集](https://huggingface.co/datasets/ZhenZeng/S-MLLMUn-data) | `S-MLLMUn/data/S-MLLMUn-data/` |
| [LLaVA 基座](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-si-hf) | `S-MLLMUn/model/base_model/llava-onevision-qwen2-7b-si-hf/` |
| [论文 Original 模型](https://huggingface.co/ZhenZeng/S-MLLMUn-llava-onevision-qwen2-7b-original) | `S-MLLMUn/model/original_model/llava-onevision-qwen2-7b-original/` |
| 本次训练适配器与 SMFA/MFA 模型 | `S-MLLMUn/model/unlearned_model/llava-onevision/split_ratio_5/` |
| 本次逐题输出与评分缓存 | `S-MLLMUn/eval_result/llava-onevision/split_ratio_5/` |

评分凭据从环境变量 `DASHSCOPE_API_KEY` 或 `OPENAI_API_KEY` 读取。本仓库不包含凭据、环境激活密钥脚本或 SSH 密钥。

## 发布范围与数据保护

- `.gitignore` 默认忽略新增产物，仅允许源代码、文档与明确列出的摘要 JSON；数据目录、权重格式、日志、JSONL、归档及图片有额外排除规则。
- 固定划分只包含人物 ID 和输入摘要；答案自检公开统计结论，去除了逐题答案；选择清单去除了题目文本。
- A 阶段 B1 清单去除原始问题及答案，仅保留索引与文本哈希；相似度检查去除最近邻评测题文本。S0 汇总去除详细激活核对数组，保留条数、最大误差和全部组统计。`A_tradeoff.png` 是唯一允许的汇总图，不含数据集图片。
- 原始完整证据仍在实验主机。部分历史文档引用的日志或数据文件不在本仓库；历史证据中的哈希对应原始文件，发布裁剪记录以 manifest 为准。
- 当前发布文件均小于 1 MiB。提交前检查：

```bash
git add <需要提交的代码或文档>
python scripts/check_publish.py
```

检查针对 Git 暂存区的实际内容，拒绝数据/模型/运行产物路径、超过 1 MiB 的文件、嵌套仓库或符号链接，以及常见凭据格式。
