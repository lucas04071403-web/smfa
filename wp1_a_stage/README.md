# WP1 A 阶段运行目录

本目录对应 `SMFA_WP1_experiment_design_v3.md`。仅评测冻结的 F1 25 人与开发保留 100 人；不运行 F2 或测试保留的新候选。

- `开工前冻结记录.md`、`protocol.json`、`FREEZE.sha256`：规则、日期及不可变输入哈希。
- `b1_selection.json`：固定种子按比例分层的 128 条 B1 问答，64 人脸、6 眼科、58 纯文本。
- `templates.json`、`template_similarity.json`：B2 与外观留出模板、相似度及语义复核；含与 v3 示例不同的原因。
- `c1_exact_guard.json`：168 个校准模块匹配 MFA，106 个固定模块匹配 SMFA。
- `operator_unit_checks.json`：r=0、λ=0/大值、硬投影与正交补、有限值和固定模块检查。
- `pilot/`：两层十条试跑；`statistics/`：128 条完整 B1 的 FP32 原始二阶矩及 84 组 S0 特征分解。
- `references/`：第 0 步生成按开发人物抽取的参照，含来源哈希。
- `candidates/配置名/`：配置、代码哈希、状态、构造耗时、逐人物缓存、逐题输出及全部开发指标。
- `selection.json`：按冻结的 C1 曲线选择 λ；若需要补强度点，先执行冻结规则，禁止外推。
- `factuality/`：A 后 3×325 条同期事实性分。输入、索引、内容哈希、API 返回模型名/时间/原始响应与重试记录均保存。
- `A_closeout.json`、`SMFA_WP1_A阶段_实验记录与结果.md`：评分漂移、人物配对 bootstrap、余量决定和最终报告。不存在这些文件时不能声称 A 已完成。
- `audit_completion.py`、`completion_audit.json`：完成后的源题/人物范围、构造代码哈希、评分输入及答案元组检查；`RESULTS.sha256`保存最终报告与关键汇总的哈希。
- `status.json`、`controller.log`：自动调度的实时状态与错误；每个候选另有独立状态。

实验 Python：`/root/anaconda3/envs/smllmun/bin/python`。

两张 GPU 各跑一个评测进程，每个进程加载一次模型并循环其候选。统计进程在 GPU 1 上并行运行，用 CPU 内存存累加矩阵，避免三份矩阵占满显存；矩阵乘法仍是 FP32，TF32 关闭。统计仅前向模型主体，不计算词表 logits。

提示模板的图像/纯文本 token 边界实测不完全相同。公共前缀仅保留逐 token 一致的部分，其余进入文本统计。FP16 激活随序列长度的微小数值差异以最大绝对差和相对 Frobenius 范数记录；按 v3 §3.6，用图像字节哈希首次出现的视觉激活构造去重矩阵，并乘样本中的精确出现次数构造重复矩阵。两者使用同一激活，避免 FP16 序列长度数值误差混入去重对照。重复视觉激活的相对 Frobenius 差须 ≤1%；异常样本对已做三层九组完整激活核对，结果见 duplicate_activation_diagnostic.json。统计每16条保存恢复检查点。

控制器顺序：首批 6 候选与统计并行 → 两张卡分别运行 C2/C3 → 必要的 C1 包围补点 → 同期交错提交 975 条评分 → 生成报告。控制器不会启动 B 阶段。失败可用人物缓存与评分 JSONL 恢复；须先查明错误，不能绕过自检或修改冻结规则。

本目录的样本清单、逐题回答、原始 API 响应、`.pt` 矩阵、日志均只留在实验机器；不要整体上传 Git。可发布源代码、脱敏汇总与实验报告。
