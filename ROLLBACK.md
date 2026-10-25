# 版本恢复

本次归档标签：`wp1-b1-complete-2026-10-25`，对应C1补点与B1完成、后续协议修订尚未实施的状态。

此前A阶段标签仍保留：`wp1-a-stage-complete-2026-10-24`。

在新目录检出该版本：

```bash
git clone git@github.com:lucas04071403-web/smfa.git
cd smfa
git switch --detach wp1-b1-complete-2026-10-25
```

已有仓库时，可保留当前工作目录，另外创建一个恢复目录：

```bash
git fetch origin --tags
git worktree add ../smfa-b1-complete wp1-b1-complete-2026-10-25
```

需要基于归档继续修改时，在恢复目录执行 `git switch -c restore-b1`。需要撤销主分支上的某次提交时，先查看 `git log --oneline`，再用 `git revert <提交哈希>` 创建撤销提交。

归档包括代码、环境依赖、设计文档、固定划分和精选结果汇总。数据集、模型、适配器、激活矩阵、逐题输出、评分缓存、日志和凭据保留在 `/root/lucas4`，Git 不能恢复这些未上传的文件。恢复代码后需核对原主机绝对路径及依赖。

`FREEZE.sha256`、`RESULTS.sha256` 及补充阶段的 `FREEZE_supplement.sha256`、`RESULTS_supplement.sha256` 记录实验主机原始文件的历史哈希；部分证据未上传，部分摘要经过裁剪。公开文件的哈希与裁剪方式以 `PUBLISH_MANIFEST.json` 为准。不要用公开裁剪后的样本清单覆盖原始冻结输入，也不要将历史状态文件直接用于恢复一个新实验。

本版本的 `B1_operator_selection.json` 按原规则选择C3。B1后讨论的图像通道+5百分点监控、改选C2和B2逐步搜索尚未实施；详见 `wp1_supplement/归档状态与后续讨论.md`。恢复此标签即恢复该修订前版本，后续修订应另立提交和标签。
