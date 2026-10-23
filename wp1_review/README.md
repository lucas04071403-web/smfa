# WP1 第 0 步补齐与 CPU 准备

固定人物划分沿用 `fixed_person_split.json`，没有重抽。seed=42；F1/F2 各 25 人；B1 为实际 RA few-shot 的 50 人；其余保留人物为开发 100、测试 200、X/H 各 300。`fixed_person_split_validation.json` 校验输入 SHA-256、不重叠、覆盖和 B1 排除。冻结前为确认输出格式读过少量原始记录的情况保留在 manifest 的 `protocol_note` 中，不将本次划分追溯表述为完全未看过任何输出。

## 已有结果的拆分

`step0_image_understanding_report.md` 给出 Original/SMFA 的人脸、眼科理解：ROUGE-L、事实性、有意义性、精确/宽松拒答率、平均回答词数。逐题结果及 cohort 在 `*_image_understanding_split.json`，汇总和输入哈希在 `existing_i_u_split_summary.json`。

文本题人物由 parquet 的题目顺序恢复，并核对题型、问题、标准答案及图片人物/类型。评分优先按 JSONL 的 index + generation SHA-256 + 题目完整元组匹配；不依赖多线程完成顺序。无缓存时用完整元组连接正式评分 JSON，遇到无法分辨的重复文本题会报错。缺少完整评分时不报告部分题目的均分。拒答率来自词法规则，不能解释为人工语义标注。

```bash
/root/anaconda3/envs/smllmun/bin/python /root/lucas4/wp1_review/prepare_wp1.py --analyze-existing
```

## LoRA 与内存构造

`memory_construct.py` 读取 epoch_3 的 MFA_multi、MFA_text、RA。遵循实测 PEFT 加载精度：首个适配器保持 FP32，随后两个先经 FP16 再回 FP32；更新和雕刻在 FP32 计算，合并后转 FP16。每次从 Original checkpoint 重新取权重，不在上一候选的 FP16 权重上回减。提供逐 tensor 构造和 `apply_memory_weights` 内存应用入口，不保存候选模型。

`memory_construct_self_check.json` 验证 274 个适配层与保存的 SMFA，以及其余未修改 tensor；`memory_construct_answer_selection.json` 冻结 **F1 开发遗忘集**按源数据顺序的前 30 题；`memory_construct_answer_check.json` 记录逐题答案对比。回答自检只依赖已有 SMFA/Original/LoRA/评测文件，不依赖两个新基线结果。

非适配张量按官方 `torch_dtype=float16` 加载后的值比较。Original 的 `image_newline` 是 FP32，保存的 SMFA 为 FP16；按加载规则转换后逐元素相等。`memory_construct_precision_correction.json` 和 `source_snapshots/` 保留修正记录与启动时源码，并通过 AST 哈希确认 LoRA 构造和答案生成函数未改变。

```bash
/root/anaconda3/envs/smllmun/bin/python /root/lucas4/wp1_review/memory_construct.py --self-check
CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/smllmun/bin/python /root/lucas4/wp1_review/memory_construct.py --verify-answers
/root/anaconda3/envs/smllmun/bin/python /root/lucas4/wp1_review/validate_preparation.py
```

## 两卡流程

`run_step0_baselines.py`：GPU 0 先完成 SMFA 前 30 题答案自检，再完整评测 Base；GPU 1 同时用仓库原版 `unlearning_methods/SMFA_sculpt.py` 执行 epoch_3、multi_k=text_k=1e20，然后评测 MFA_unsculpted。两条流程各自评测后用原有 `reproduce_smfa/score.py`、qwen-plus、同一提示词和每模型 4 个 workers 打事实性/有意义性分。完成后自动重算拆分汇总，纳入 Base/MFA。

官方雕刻脚本未修改。它只打印掩码数，因此启动器解析**本次**雕刻的两条日志并要求均为零、参数总数为正，另检查 checkpoint 分片存在，再写 `step0_baselines/MFA_unsculpted_zero_mask.json`。不要求官方未提供的 `sculpt_stats.json`。

启动器自动读取已配置的 Conda API 激活脚本，只保存凭据来源，不输出密钥。仅在本任务进程中清除继承的代理变量，使用已实测可连接的 DashScope 直连；宿主机代理配置不变。服务凭据/模型列表验证记录为 `step0_baselines/scoring_preflight.json`。

```bash
python /root/lucas4/wp1_review/run_step0_baselines.py --launch
# 中断后复用已完成阶段、逐人生成缓存、逐题评分缓存和已记录耗时：
python /root/lucas4/wp1_review/run_step0_baselines.py --launch --resume
# 等两张卡各有至少 32768 MiB 空闲后启动：
python /root/lucas4/wp1_review/run_step0_baselines.py --launch --wait-for-gpus
```

当前状态以 `step0_baselines/status.json` 为准，外层日志是 `step0_baselines/runner.log`，每阶段有单独日志。结束后 `step0_baselines/timings.json` 汇总各模型评测、评分的墙钟秒数；包括加载/ROUGE 或 API 等待/重试，不把雕刻和 SMFA 自检混入评测耗时。重跑会累计已记录的尝试；进程异常退出而未记录的时间会标为 `incomplete_timing`，不能据此声称全程精确耗时。

2026-10-23 评分恢复：两组评测已结束；MFA 首次事实性评分有 3 条空分数，导致该评分子进程退出。`recover_step0_scoring.py` 使用**未修改的**原评分脚本、模型、提示词及缓存，只补缺失评分。第 2 次尝试补齐两条，第 3 次尝试补齐剩余一条，随后开始有意义性评分。重试会保留失败尝试及其耗时，最多新增 5 次尝试。评分后逐条核验索引、生成内容摘要、分数范围及导出文件；最终重新生成四个模型的拆分统计。

恢复期间同时查看 `step0_baselines/scoring_recovery.json` 和原 `status.json`：原进程仍负责 Base，恢复进程负责 MFA；恢复进程拿到原进程释放的锁后才合并状态与耗时，避免互相覆盖。`progress_snapshot.json` 是带检查时间的快照。此阶段不要另启 `--resume`；已知 CPU 自检精度审计的源码修正也会触发原启动器的严格指纹检查，恢复程序保留并验证该修正记录，不改写启动时的源码指纹。评分总耗时累加各次实际执行时间，不包含失败后等待恢复的空闲时间。

2026-10-22 已按用户要求清理两张 GPU 原有计算进程，日志为 `gpu_cleanup_*.json`。原版雕刻同时持有三份稠密更新、两份 mask 和两份雕刻副本，峰值超过本机 128 GiB RAM；本轮为该进程启用独立 64 GiB 临时 swap。`sculpt_swap_guard.py` 在目标雕刻进程退出后自动 swapoff 并删除文件，状态和日志在 `step0_baselines/official_sculpt_swap_*.json`、`sculpt_swap_guard.log`。这只改变可用内存容量，不改变模型计算或官方脚本。
