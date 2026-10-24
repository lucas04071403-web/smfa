# 版本恢复

本次 A 阶段归档标签：`wp1-a-stage-complete-2026-10-24`。

在新目录检出该版本：

```bash
git clone git@github.com:lucas04071403-web/smfa.git
cd smfa
git switch --detach wp1-a-stage-complete-2026-10-24
```

已有仓库时，可保留当前工作目录，另外创建一个恢复目录：

```bash
git fetch origin --tags
git worktree add ../smfa-a-stage wp1-a-stage-complete-2026-10-24
```

需要基于归档继续修改时，在恢复目录执行 `git switch -c restore-a-stage`。需要撤销主分支上的某次提交时，先查看 `git log --oneline`，再用 `git revert <提交哈希>` 创建撤销提交。

归档包括代码、环境依赖、设计文档、固定划分和精选结果汇总。数据集、模型、适配器、激活矩阵、逐题输出、评分缓存、日志和凭据保留在 `/root/lucas4`，Git 不能恢复这些未上传的文件。恢复代码后需核对原主机绝对路径及依赖。

`FREEZE.sha256` 和 `RESULTS.sha256` 记录实验主机原始文件的历史哈希；部分证据未上传，部分摘要经过裁剪。公开文件的哈希与裁剪方式以 `PUBLISH_MANIFEST.json` 为准。不要用公开裁剪后的样本清单覆盖原始冻结输入，也不要将历史状态文件直接用于恢复一个新实验。
