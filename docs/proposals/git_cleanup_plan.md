# Git 仓库分支/标签/Stash 清理审查报告

> **声明**：本报告仅基于只读检查生成，**未执行任何破坏性 Git 操作**（如 `git branch -D`、`git push --delete`、`git remote set-url`、`git stash drop` 等）。
> 所有涉及仓库状态变更的操作都需要用户确认后再执行。

---

## 1. 当前仓库快照

### 1.1 分支与远程

```text
$ git branch -a
* main
  remotes/origin/HEAD -> origin/main
  remotes/origin/main

$ git remote -v
origin  https://adjcjh777:gho_***REDACTED***@github.com/adjcjh777/motionflow-multiview-kimiswarm.git (fetch)
origin  https://adjcjh777:gho_***REDACTED***@github.com/adjcjh777/motionflow-multiview-kimiswarm.git (push)
```

| 项目 | 结果 |
|------|------|
| 本地分支 | **仅 `main` 一条** |
| 远程分支 | **仅 `origin/main` 一条** |
| 工作树（worktree） | 1 个：`.worktrees/v18_deformable_attention_baseline`，但当前指向 `main` |
| 远程仓库 | `origin` 指向 GitHub，URL 中嵌入 token |

### 1.2 当前工作区状态

存在**未提交修改**，建议先提交到临时分支或当前分支，**不要 force push**。

**已修改（tracked）：**

- `AGENTS.md`
- `docs/handoff_qwen3.8max.md`
- `docs/paper_draft_icra_cvpr_2027.md`
- `docs/results_true_gt_h36m.md`
- `docs/status_dashboard_v2.md`
- `scripts/launch_v25_true_gt_v2_medium_a800.sh`
- `scripts/launch_v86_sparse_cross_domain_v2_medium_a800.sh`
- `scripts/run_v25_true_gt_v2_medium_a800.sh`

**未跟踪（untracked）：**

- `configs/ablations/v86_sparse_cross_domain_v2_medium_a800.yaml`
- `docs/proposals/circular_label_diagnosis.md`
- `docs/proposals/cvpr2027_milestone.md`
- `docs/proposals/git_cleanup_plan.md`
- `patches/`（整个目录，是 stash patch 备份）
- `scripts/diagnose_circular_h36m_labels.py`
- `scripts/launch_v85_dlt_fallback_after_v86.sh`
- `scripts/launch_v86_after_v25_a800.sh`
- `scripts/monitor_v85_dlt_fallback.py`
- `scripts/run_v21_neural_ba_true_gt_v2_smoke_local_4090.sh`
- `scripts/run_v29_hierarchical_true_gt_v2_smoke_local_4090.sh`
- `scripts/run_v37_self_critique_true_gt_v2_smoke_local_4090.sh`

### 1.3 Stash

- 当前共有 **45 条 stash**（`stash@{0}` ~ `stash@{44}`）。
- 已全部备份为 patch 文件到 `patches/stashes/`。
- Stash 内容大多来自历史实验分支的 WIP，但**对应分支本身已不存在**（只在 stash message 中保留分支名，如 `v48-domain`、`v47-temporal`、`swarm/v21_neural_ba_diagnosis` 等）。

### 1.4 本地标签

- `v25_local_baseline_monitor_commit`
- `v25_local_baseline_monitor_v1`

这两个是轻量标签，目前**仅存在于本地**，未推送到远程（`git ls-remote --tags origin` 无返回）。

---

## 2. 分支分类

> 说明：以下基于 `git branch -a` 与 `git for-each-ref refs/` 的结果进行分类。

### 2.1 主分支 / 活跃开发分支（保留）

| 分支 | 状态 | 操作建议 |
|------|------|----------|
| `main` | 当前 HEAD，最新提交 `bbf8895` | **保留**，是主开发分支 |

### 2.2 已合并到主分支的死分支（可删除）

**无。**

`git branch -a` 只显示 `main` 和 `origin/main`，没有其它本地或远程分支。

### 2.3 旧实验分支（可删除）

**无独立的本地/远程分支。**

实验中提到的 `v31`–`v79` 等历史 feature 分支目前**不存在为 Git 分支**，它们只以 stash 消息的形式残留在 45 条 stash 中。例如：

- `v48-domain`
- `v47-temporal`
- `swarm/v21_neural_ba_diagnosis`
- `swarm/smpl_prior_fusion_experiment`
- `feat/iter17-*`
- `feat/iter-next-*`
- `feature/mixed-dataset-balanced-sampling`
- `run/visibility-uncertainty-v1`
- 等

这些内容已经以 patch 形式备份在 `patches/stashes/`，可以**删除原始 stash**，但必须先确认 patch 备份完整。

### 2.4 需要人工确认的分支 / 标签

| 名称 | 类型 | 说明 | 建议 |
|------|------|------|------|
| `.worktrees/v18_deformable_attention_baseline` | 工作树 | 目录名暗示 `v18_deformable_attention_baseline` 分支，但实际指向 `main` | 确认是否还需该工作树；若不需要，可 `git worktree remove` |
| `v25_local_baseline_monitor_commit` | 轻量标签 | 本地临时监控标签 | 确认是否还需保留 |
| `v25_local_baseline_monitor_v1` | 轻量标签 | 本地临时监控标签 | 确认是否还需保留 |

---

## 3. Remote URL 中的 Token/PAT 检查

当前 remote URL 包含 GitHub OAuth token：

```text
https://adjcjh777:gho_***REDACTED***@github.com/...
```

### 风险

- Token 明文写入 `.git/config`，一旦泄露即被滥用。
- 当前 token 前缀 `gho_` 为 GitHub OAuth token。

### 建议处理（需用户确认后执行）

```bash
# 1. 查看当前 URL
git remote -v

# 2. 替换为无 token 的 HTTPS URL
#    推荐方式 A：普通 HTTPS（推送时凭 git-credential 管理密码/token）
git remote set-url origin https://github.com/adjcjh777/motionflow-multiview-kimiswarm.git

#    推荐方式 B：SSH
git remote set-url origin git@github.com:adjcjh777/motionflow-multiview-kimiswarm.git

# 3. 验证
git remote -v
```

### 额外必要操作

1. 到 GitHub Settings → Developer settings → Personal access tokens → Tokens (classic) **撤销该 token**。
2. 清理 shell 历史、tmux/screen 日志、`.bash_history`、`.zsh_history` 中可能包含该 token 的记录。
3. 检查是否有 `.git/config` 的备份或历史版本也包含该 token。

**本报告不会执行上述修改。**

---

## 4. 清理计划

### 4.1 可安全删除 / 清理的项目

| 项目 | 数量 | 操作 | 前提条件 |
|------|------|------|----------|
| 已合并分支 | 0 | 无 | — |
| 旧实验分支 | 0 | 无 | — |
| Stash 条目 | 45 | 可清空 stash | 已备份到 `patches/stashes/`（已确认） |
| 本地轻量标签 `v25_local_baseline_monitor_*` | 2 | 删除本地标签 | 确认不再需要 |
| `.worktrees/v18_deformable_attention_baseline` | 1 | 移除工作树 | 确认工作树内容已无用 |

### 4.2 需要保留的项目

| 项目 | 原因 |
|------|------|
| `main` | 主开发分支 |
| `patches/stashes/*.patch` | 是 45 条 stash 的唯一备份，删除前必须保留 |

### 4.3 需要人工确认的项目

| 项目 | 问题 |
|------|------|
| 45 条 stash 的 patch 备份是否完整 | 检查 `patches/stashes/` 是否 45 个文件都能正常 `git apply --check` |
| 两个 `v25_local_baseline_monitor_*` 标签是否有用 | 是否用于脚本/监控/版本标记 |
| `.worktrees/v18_deformable_attention_baseline` 是否还在使用 | 是否有未保存的实验代码 |
| 未提交文件中的 scripts/configs 是否应提交 | 是否属于当前主线的必要变更 |

---

## 5. 当前未提交变更的处理建议

由于存在大量未提交的 docs/ 和 scripts/ 变更，建议先提交到当前分支或一个临时分支，避免清理 stash 时意外覆盖。

### 推荐操作（需用户确认）

```bash
# 方案 A：提交到当前 main（如果不介意当前 main 上有新提交）
git add docs/ scripts/diagnose_circular_h36m_labels.py \
          scripts/launch_v85_dlt_fallback_after_v86.sh \
          scripts/launch_v86_after_v25_a800.sh \
          scripts/monitor_v85_dlt_fallback.py \
          scripts/run_v21_neural_ba_true_gt_v2_smoke_local_4090.sh \
          scripts/run_v29_hierarchical_true_gt_v2_smoke_local_4090.sh \
          scripts/run_v37_self_critique_true_gt_v2_smoke_local_4090.sh \
          configs/ablations/v86_sparse_cross_domain_v2_medium_a800.yaml \
          docs/proposals/circular_label_diagnosis.md \
          docs/proposals/cvpr2027_milestone.md \
          docs/proposals/git_cleanup_plan.md
git commit -m "WIP: current docs/scripts/config changes before cleanup"

# 方案 B：创建临时分支保存
git checkout -b tmp/cleanup-snapshot-$(date +%Y%m%d)
# 然后 add + commit 同上
```

**不要 force push。**

---

## 6. 后续可执行命令（待用户确认）

```bash
# 1. 移除 remote 中的 token（任选一种）
git remote set-url origin https://github.com/adjcjh777/motionflow-multiview-kimiswarm.git
# 或
git remote set-url origin git@github.com:adjcjh777/motionflow-multiview-kimiswarm.git

# 2. 删除两个本地标签
git tag -d v25_local_baseline_monitor_commit
git tag -d v25_local_baseline_monitor_v1

# 3. 移除不再使用的工作树（如果确认）
git worktree remove .worktrees/v18_deformable_attention_baseline

# 4. 清空所有 stash（前提：确认 patches/stashes/ 备份完整）
git stash clear

# 5. 验证
git remote -v
git branch -a
git tag
git stash list
```

---

## 7. 总结

| 项目 | 数量 / 状态 |
|------|-------------|
| 本地分支总数 | **1**（`main`） |
| 远程分支总数 | **1**（`origin/main`） |
| 可安全删除的本地分支 | **0** |
| 需要人工确认的分支/标签/工作树 | **3**（1 个工作树、2 个本地标签） |
| Stash 总数 | **45**（已备份到 `patches/stashes/`） |
| 可删除的 Stash 条目 | **45**（需先确认 patch 备份完整） |
| Remote URL 是否包含 token/PAT | **是**，包含 `gho_` 开头的 GitHub OAuth token |
| 未提交变更 | 有，建议先提交到临时分支 |

---

## 8. 检查清单（执行清理前）

- [ ] 已确认 `patches/stashes/` 中 45 个 patch 文件完整且能 `git apply --check` 通过。
- [ ] 已确认两个 `v25_local_baseline_monitor_*` 标签不再需要。
- [ ] 已确认 `.worktrees/v18_deformable_attention_baseline` 没有未保存的工作。
- [ ] 已备份当前未提交变更（建议先 commit 到 `tmp/cleanup-snapshot` 分支）。
- [ ] 已撤销 GitHub 上对应的个人访问 token。
- [ ] 已清理 shell 历史、日志中的 token 明文。
