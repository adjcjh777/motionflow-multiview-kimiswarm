# GitHub 仓库清理执行计划

> ⚠️ **本计划需要用户确认后才能执行。**
> 本文档仅用于规划和准备命令清单，**不执行任何实际的 Git 状态变更**（不运行 commit、remote set-url、branch delete、push 等）。

---

## 执行状态（2026-08-13）

- ✅ **remote URL 中的 token 已移除**：当前 remote URL 为不含 token 的 HTTPS URL。
- ✅ **旧工作树已删除**：`.worktrees/v18_deformable_attention_baseline` 已移除。
- ✅ **本地轻量标签已删除**：`v25_local_baseline_monitor_commit`、`v25_local_baseline_monitor_v1` 已移除。
- ✅ **main 分支已 push**：本地 `main`（commit `8aee08c` 或更新）已 push 到 GitHub；GitHub 仓库现在包含清理后的状态。
- ⏳ **`patches/stashes/` 中 45 个 stash patch 备份仍保留**，待后续审计或清理。

---

## 1. 当前状态总结

### 1.1 分支状态

```text
$ git branch -a
* main
  remotes/origin/HEAD -> origin/main
  remotes/origin/main
```

- 本地分支：**仅 `main`**
- 远程分支：**仅 `origin/main`**

### 1.2 Stash 状态

- 当前共有 **45 条 stash**（`stash@{0}` ~ `stash@{44}`）。
- 已全部备份为 patch 文件到 `patches/stashes/`（共 45 个 `.patch` 文件）。
- 由于对应的历史实验分支多数已不存在，这些 stash 仅作为历史备份保留。

### 1.3 Remote URL 状态

```text
$ git remote -v
origin  https://github.com/adjcjh777/motionflow-multiview-kimiswarm.git (fetch)
origin  https://github.com/adjcjh777/motionflow-multiview-kimiswarm.git (push)
```

- Remote `origin` 的 URL 中**已移除 GitHub OAuth token**，当前为普通 HTTPS URL。

### 1.4 未提交变更

当前工作区存在未提交的 `docs/`、`scripts/`、`configs/` 等变更：

**已修改（modified）：**

- `AGENTS.md`
- `docs/handoff_qwen3.8max.md`
- `docs/paper_draft_icra_cvpr_2027.md`
- `docs/results_true_gt_h36m.md`
- `docs/status_dashboard_v2.md`
- `motionflow_mv/fusion/epipolar_attention_bias.py`
- `motionflow_mv/fusion/omniview_fusion_v5.py`
- `motionflow_mv/fusion/principal_point_correction.py`
- `scripts/launch_v25_true_gt_v2_medium_a800.sh`
- `scripts/launch_v86_sparse_cross_domain_v2_medium_a800.sh`
- `scripts/run_v25_true_gt_v2_medium_a800.sh`

**未跟踪（untracked）：**

- `configs/ablations/v86_sparse_cross_domain_v2_medium_a800.yaml`
- `docs/proposals/circular_label_diagnosis.md`
- `docs/proposals/cvpr2027_milestone.md`
- `docs/proposals/git_cleanup_plan.md`
- `docs/proposals/git_cleanup_execution_plan.md`
- `patches/`（整个目录，stash patch 备份）
- `scripts/diagnose_circular_h36m_labels.py`
- `scripts/launch_v85_dlt_fallback_after_v86.sh`
- `scripts/launch_v86_after_v25_a800.sh`
- `scripts/monitor_v85_dlt_fallback.py`
- `scripts/run_v21_neural_ba_true_gt_v2_smoke_local_4090.sh`
- `scripts/run_v29_hierarchical_true_gt_v2_smoke_local_4090.sh`
- `scripts/run_v37_self_critique_true_gt_v2_smoke_local_4090.sh`
- `scripts/run_v86_sparse_cross_domain_v2_medium_a800.sh`

### 1.5 工作树（worktree）状态

```text
$ git worktree list
D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm d2ed343 [main]
```

- `.worktrees/v18_deformable_attention_baseline` 旧工作树**已删除**；当前仅剩主工作目录。

### 1.6 本地轻量标签

- ~~`v25_local_baseline_monitor_commit`~~ **已删除**
- ~~`v25_local_baseline_monitor_v1`~~ **已删除**

### 1.7 Push 结果

- 本地 `main` 分支已 push 到 GitHub，当前仓库 HEAD 指向 commit `8aee08c`（或更新）。
- GitHub 远端 `origin/main` 与本地 `main` 一致。

---

## 2. 清理步骤清单（按顺序）

### 步骤 1：保存当前未提交变更

目标：将当前工作区的未提交变更保存到一个临时分支或当前 `main` 分支的 commit 中，避免后续清理时丢失。

推荐方案 A：提交到临时分支（推荐，不污染 main 历史）。

```bash
# 确认当前在 main 分支
git branch --show-current

# 创建临时分支
git checkout -b tmp/cleanup-snapshot-$(date +%Y%m%d)

# 添加所有未提交变更
# 注意：patches/ 是 stash 备份目录，如果希望保留在临时分支中，可以一并添加；
#       如果希望保持为未跟踪文件，则不添加 patches/。
git add docs/ scripts/ configs/ motionflow_mv/
git add docs/proposals/circular_label_diagnosis.md \
       docs/proposals/cvpr2027_milestone.md \
       docs/proposals/git_cleanup_plan.md \
       docs/proposals/git_cleanup_execution_plan.md

# 提交
git commit -m "WIP: snapshot of uncommitted docs/scripts/config changes before cleanup"

# 切回 main
git checkout main
```

推荐方案 B：直接提交到当前 `main` 分支（如果确认这些变更属于当前主线）。

```bash
git add docs/ scripts/ configs/ motionflow_mv/
git add docs/proposals/circular_label_diagnosis.md \
       docs/proposals/cvpr2027_milestone.md \
       docs/proposals/git_cleanup_plan.md \
       docs/proposals/git_cleanup_execution_plan.md

git commit -m "WIP: current docs/scripts/config changes before cleanup"
```

验证：

```bash
git status --short
git log --oneline -3
```

---

### 步骤 2：移除 remote URL 中的 token

目标：将包含 `gho_` token 的 HTTPS URL 替换为普通 HTTPS 或 SSH URL。

```bash
# 查看当前 remote URL
git remote -v

# 方式 A：普通 HTTPS（推荐，使用 git-credential 管理后续认证）
git remote set-url origin https://github.com/adjcjh777/motionflow-multiview-kimiswarm.git

# 方式 B：SSH（需要本地已配置 SSH key 并与 GitHub 关联）
git remote set-url origin git@github.com:adjcjh777/motionflow-multiview-kimiswarm.git
```

---

### 步骤 3：验证 remote URL 不再包含 token

```bash
git remote -v
```

预期输出（无 token 部分）：

```text
origin  https://github.com/adjcjh777/motionflow-multiview-kimiswarm.git (fetch)
origin  https://github.com/adjcjh777/motionflow-multiview-kimiswarm.git (push)
```

或 SSH 形式：

```text
origin  git@github.com:adjcjh777/motionflow-multiview-kimiswarm.git (fetch)
origin  git@github.com:adjcjh777/motionflow-multiview-kimiswarm.git (push)
```

额外建议：

```bash
# 确认 .git/config 中不再包含 token
grep -n "gho_" .git/config || echo "Token not found in .git/config"
```

---

### 步骤 4：删除备份的 stash patches 中的旧文件（如果确定不再需要）

当前 `patches/stashes/` 下共有 45 个 `.patch` 文件，是 45 条 stash 的备份。

```bash
# 1. 先验证 patch 文件数量和完整性
ls -1 patches/stashes/*.patch | wc -l
ls -la patches/stashes/

# 2. 可选：验证 patch 是否能正常 apply（只检查，不应用）
for p in patches/stashes/*.patch; do
    echo "Checking: $p"
    git apply --check "$p" || echo "FAILED: $p"
done

# 3. 如果确认不再需要，删除整个 patches/stashes/ 目录
rm -rf patches/stashes/

# 4. 验证删除结果
ls -la patches/
```

---

### 步骤 5：清理 `.worktrees/` 中的旧工作树（如果不再需要）

```bash
# 查看工作树列表
git worktree list

# 移除指定工作树
git worktree remove .worktrees/v18_deformable_attention_baseline

# 如果目录残留，手动删除（仅在 git worktree remove 未完全清理时使用）
rm -rf .worktrees/v18_deformable_attention_baseline

# 验证
git worktree list
ls -la .worktrees/
```

---

### 步骤 6：可选 — 删除本地轻量标签 `v25_local_baseline_monitor_*`

```bash
# 查看本地标签
git tag -l 'v25_local_baseline_monitor_*'

# 删除本地标签
git tag -d v25_local_baseline_monitor_commit
git tag -d v25_local_baseline_monitor_v1

# 验证
git tag
```

---

### 步骤 7：push 当前 main 分支到 remote（已完成）

状态：✅ 已执行。本地 `main` 已 push 到 GitHub，远端 `origin/main` 当前指向 commit `8aee08c`（或更新）。

如果步骤 1 中在 `main` 分支上创建了新的 commit，并且希望同步到远程：

```bash
# 先确认 remote URL 已安全
git remote -v

# 拉取远端最新变更，避免冲突
git pull origin main

# 推送当前 main
git push origin main
```

如果步骤 1 使用的是临时分支方案，且希望将临时分支也推送到远端备份：

```bash
git checkout tmp/cleanup-snapshot-$(date +%Y%m%d)
git push -u origin tmp/cleanup-snapshot-$(date +%Y%m%d)
```

---

## 3. 完整命令速查表

```bash
# ============================================================
# 0. 前置检查（执行前务必运行）
# ============================================================
git status --short
git branch -a
git remote -v
git stash list
git tag
git worktree list

# ============================================================
# 1. 保存当前未提交变更（二选一）
# ============================================================
# 方案 A：临时分支
git checkout -b tmp/cleanup-snapshot-$(date +%Y%m%d)
git add docs/ scripts/ configs/ motionflow_mv/
git add docs/proposals/circular_label_diagnosis.md \
       docs/proposals/cvpr2027_milestone.md \
       docs/proposals/git_cleanup_plan.md \
       docs/proposals/git_cleanup_execution_plan.md
git commit -m "WIP: snapshot of uncommitted docs/scripts/config changes before cleanup"
git checkout main

# 方案 B：直接提交到 main
git add docs/ scripts/ configs/ motionflow_mv/
git add docs/proposals/circular_label_diagnosis.md \
       docs/proposals/cvpr2027_milestone.md \
       docs/proposals/git_cleanup_plan.md \
       docs/proposals/git_cleanup_execution_plan.md
git commit -m "WIP: current docs/scripts/config changes before cleanup"

# ============================================================
# 2. 移除 remote URL 中的 token
# ============================================================
git remote set-url origin https://github.com/adjcjh777/motionflow-multiview-kimiswarm.git
# 或
git remote set-url origin git@github.com:adjcjh777/motionflow-multiview-kimiswarm.git

# ============================================================
# 3. 验证 remote URL
# ============================================================
git remote -v
grep -n "gho_" .git/config || echo "Token not found in .git/config"

# ============================================================
# 4. 删除 stash patch 备份（如果确认不再需要）
# ============================================================
ls -1 patches/stashes/*.patch | wc -l
rm -rf patches/stashes/
ls -la patches/

# ============================================================
# 5. 清理旧工作树
# ============================================================
git worktree remove .worktrees/v18_deformable_attention_baseline
rm -rf .worktrees/v18_deformable_attention_baseline

# ============================================================
# 6. 删除本地轻量标签（可选）
# ============================================================
git tag -d v25_local_baseline_monitor_commit
git tag -d v25_local_baseline_monitor_v1

# ============================================================
# 7. 推送 main（如果需要）
# ============================================================
git pull origin main
git push origin main

# ============================================================
# 8. 最终验证
# ============================================================
git status --short
git branch -a
git remote -v
git tag
git stash list
git worktree list
```

---

## 4. 回滚方案

| 操作 | 回滚方法 |
|------|----------|
| 误删/误改 remote URL | `git remote set-url origin <原始 URL>` 或手动编辑 `.git/config` |
| 误删本地标签 | 如果已推送到远程，可从远端重新拉取；否则无法恢复，需谨慎 |
| 误删 `patches/stashes/` | 如果已删除且未备份，**不可恢复**；建议删除前额外复制一份到仓库外 |
| 误删工作树 | 如果工作树内代码已提交到某分支，可从对应分支恢复；否则可能丢失 |
| 未提交变更丢失 | 步骤 1 完成后，未提交内容已作为 commit 保存到临时分支或 main，可随时 checkout 恢复 |
| 误 commit 到 main | 若未 push，可 `git reset --soft HEAD~1` 回退；若已 push，需用 `git revert` 而非 force push |

---

## 5. 风险说明

| 操作 | 可逆性 | 风险等级 | 说明 |
|------|--------|----------|------|
| 保存未提交变更到临时分支/commit | 可逆 | 低 | 只是新增 commit，不会丢失数据 |
| 修改 remote URL | 可逆 | 低 | 可随时 `git remote set-url` 恢复；但需确保替换后仍能正常 push/pull |
| 删除 `patches/stashes/` | **不可逆** | 高 | 这是 45 条 stash 的唯一备份，删除前务必确认不再需要 |
| 删除 `.worktrees/` 中的工作树 | 视情况而定 | 中 | 如果工作树内包含未提交且未备份的代码，删除后将丢失 |
| 删除本地轻量标签 | **不可逆**（若未推送） | 中 | 仅本地存在，删除前确认不再需要 |
| push 到 remote | 不可撤回 | 中 | push 前确认 commit 内容正确，避免 force push |

**特别注意事项：**

1. **Token  revocation：** 替换 remote URL 后，建议到 GitHub Settings → Developer settings → Personal access tokens → Tokens (classic) 中撤销该 `gho_` token，并清理 `.bash_history`、`.zsh_history`、tmux/screen 日志等可能包含 token 明文的记录。
2. **SSH 可用性：** 如果选择 SSH URL，请确保本地已配置 GitHub SSH key，否则后续 `git push` 会失败。
3. **45 条 stash：** 这些 stash 已经备份为 patch 文件，原始 stash 条目可以在确认 patch 备份完整后清理。

---

## 6. 总结

本计划针对 `D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm` 仓库的当前状态，提出了以下清理步骤：

1. **保存未提交变更**：将当前 `docs/`、`scripts/`、`configs/` 等未提交变更保存到临时分支或 `main` 的 commit 中。
2. **移除 remote URL 中的 token**：将包含 `gho_` 的 OAuth token URL 替换为普通 HTTPS 或 SSH URL。
3. **验证 remote URL 安全**：确认 `.git/config` 和 `git remote -v` 不再包含 token。
4. **删除 stash patch 备份**：在确认不再需要 `patches/stashes/` 中的 45 个 patch 文件后删除。
5. **清理旧工作树**：移除 `.worktrees/v18_deformable_attention_baseline`。
6. **删除本地轻量标签**：可选删除 `v25_local_baseline_monitor_commit` 和 `v25_local_baseline_monitor_v1`。
7. **push main**：将清理后的 `main` 分支同步到 remote（如需）。

> ⚠️ **再次强调：本计划需要用户确认后才能执行。**
