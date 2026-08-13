# MotionFlow-MultiView 接力手册（qwen3.8max）

> **目标**：把 H36M 评估协议迁移到非循环的 `h36m_true_gt_v2`，验证 v85 稀疏视角方案，补齐 v86 消融与 v2 leaderboard，向 CVPR 2027 可发表标准推进。  
> **本次刷新时间**：2026-08-13 ~01:15 UTC  
> **本地仓库**：`D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm`  
> **A800 训练仓库**：`a800-D:/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`  
> **GPU 铁律**：A800 仅 GPU 6/7 可用于本项目；GPU 0–5 严禁使用。

---

## 0. 最新状态（2026-08-13 ~00:41 UTC）

- **v25 true-GT v2 medium 训练已完成。** A800 GPU 6，tmux session `v25_true_gt_v2_medium_a800`。early stopping @ epoch 6，best val MPJPE **31.41 mm**，checkpoint 已保存为 `outputs/ablations/v25_true_gt_v2_medium_a800.pth`。test 评估待做。
- **v86 no-count-embedding ablation 已在 A800 GPU 6 启动。** tmux session `v86_no_count_embedding`，脚本 `scripts/run_v86_no_count_embedding_medium_a800_gpuX.sh`，使用 v2 数据协议 `configs/splits/h36m_true_gt_v2_standard.yaml`。结果待出。
- **v85 random-view-dropout 训练已完成。** checkpoint 已创建 symlink `outputs/ablations/v85_random_view_dropout_medium_a800.pth -> ..._final.pth`，best val **31.42 mm**。
- **v85 DLT-fallback 可变视角评估看守器已启动。** `scripts/launch_v85_dlt_fallback_after_v86.sh` 正在等待 v86 训练完成后，自动在首个可用 GPU（6 或 7）上运行 v85 DLT-fallback 评估。结果待出。
- **GPU 7 当前被外部项目占用**（约 12 GB 显存）；GPU 6 当前运行 v86。
- **磁盘 `/mnt/nvme0n1p1` 约 98% 满（~72 GB free）。**

### 当前在飞任务

| 机器 | GPU | 任务 | 日志 / 输出 | 状态 | 说明 |
|------|-----|------|-------------|------|------|
| A800-D | 6 | v25 true-GT v2 medium training | `outputs/ablations/v25_true_gt_v2_medium_a800.log` | **DONE** | tmux `v25_true_gt_v2_medium_a800`；early-stop @ epoch 6；best val **31.41 mm**；checkpoint `.pth` 已落盘。 |
| A800-D | 6 | v86 no-count-embedding ablation | `outputs/ablations/v86_no_count_embedding_medium_a800.log` | **RUNNING** | tmux `v86_no_count_embedding`；v2 协议；结果待出。 |
| A800-D | 6/7 (post-v86) | v85 DLT-fallback 可变视角评估看守器 | `outputs/launch_v85_dlt_fallback_after_v86.log` | **QUEUED** | 等待 v86 完成后自动启动 v85 DLT-fallback 评估；结果待出。 |
| A800-D | 7 | 外项目进程 | — | **OCCUPIED** | 约 12 GB；**禁止 kill**，禁止启动新任务。 |

---

## 0. 执行摘要

- **v85 random-view-dropout 训练已完成**。A800 GPU 7 训练结束，best val **31.42 mm**，checkpoint 已落盘。
- **v85 可变视角 no-fallback 评估已完成**。k=2 与 k=3 仍然灾难性，说明仅靠 random view dropout 无法解决稀疏视角问题：
  - k=2：S9 **2310.27 mm**，S11 **2308.80 mm**
  - k=3：S9 **1119.45 mm**，S11 **1118.18 mm**
  - k=4：S9 **83.52 mm**，S11 **77.07 mm**
- **v85 DLT-fallback 可变视角评估已终止且未产生输出**。监控显示 PID `2269984` 在约 29 分钟后消失，预期的 JSON/CSV 未生成。redirect log 为空，nohup log 仅显示 `Terminated`，原因不明（可能是外部 kill 或 OOM）。在重新运行前，v25/v81/v82 的 DLT-fallback 基线仍是最新可靠参考：S9 58.18/33.32/116.98 mm，S11 49.35/25.28/110.58 mm。结合 v85 no-fallback 结果，可确认 random view dropout 不能解决 k<4 灾难性失败，DLT-fallback 仍是 k<4 的可靠选择。
- **v25 true-GT v2 medium 训练已启动**。A800 GPU 6，tmux session: `v25_true_gt_v2_medium_a800`。
- **v86 no-count-embedding ablation 已由 A800 看守器接管**。看守器 `scripts/launch_v86_after_v25_a800.sh`（PID `2337615`）正在 A800 上等待 v25 完成后自动启动 v86；无需手动 `nohup`。
- **h36m_true_gt_v2 数据已同步到 A800**。约 **625 MB**，路径 `data/h36m_true_gt_v2/`；对应 manifest `configs/splits/h36m_true_gt_v2_standard.yaml`。
- **v25/v86 v2 启动脚本已同步到 A800**。见 `scripts/run_v25_true_gt_v2_medium_a800.sh` 与 `scripts/run_v86_no_count_embedding_medium_a800_gpu6.sh`。
- **A800 磁盘已清理**。从约 99% 降到 **98%**，剩余约 **73 GB**。
- **GPU 7 被外项目占用**。LuxTTS / Mega-ASR / ComfyUI 占用了 GPU 7，违反项目 GPU 策略；**禁止 kill**，但在此 cleared 之前不得在 GPU 7 启动新任务。
- **本地 v37 self-critique v2 smoke 正在运行**。RTX 4090，用于快速验证 self-critique 思路。
- **GitHub / 本地仓库清理已完成**。remote URL 中的 token 已移除（当前为 `https://github.com/adjcjh777/motionflow-multiview-kimiswarm.git`）；旧工作树 `.worktrees/v18_deformable_attention_baseline` 已删除；本地轻量标签 `v25_local_baseline_monitor_commit` 和 `v25_local_baseline_monitor_v1` 已删除；`main` 已 push 到 GitHub（commit `d2ed343`）。`patches/stashes/` 中 45 个 stash patch 备份仍保留，待后续审计。

---

## 1. 在飞任务与状态

| 机器 | GPU | 任务 | 日志 / 输出 | 状态 | 说明 |
|------|-----|------|-------------|------|------|
| A800-D | 6 | v25 true-GT v2 medium training | `outputs/ablations/v25_true_gt_v2_medium_a800.log` | **DONE** | tmux session `v25_true_gt_v2_medium_a800`；early-stop @ epoch 6；best val **31.41 mm**；checkpoint `.pth` 已落盘。test 待测。 |
| A800-D | 6 | v86 no-count-embedding ablation | `outputs/ablations/v86_no_count_embedding_medium_a800.log` | **RUNNING** | tmux session `v86_no_count_embedding`；v2 协议；结果待出。 |
| A800-D | 6/7 (post-v86) | v85 DLT-fallback 可变视角评估看守器 | `outputs/launch_v85_dlt_fallback_after_v86.log` | **RUNNING (watcher)** | `scripts/launch_v85_dlt_fallback_after_v86.sh` 等待 v86 完成后自动启动 v85 DLT-fallback 评估；v86 完成前不占用 GPU 6/7。 |
| A800-D | 7 | 外项目进程（LuxTTS / Mega-ASR / ComfyUI） | — | **OCCUPIED** | 约 12 GB；违反 GPU 策略；**禁止 kill**，禁止启动新任务。 |
| Local | 0 | v37 self-critique v2 smoke | `outputs/v37_self_critique_v2_smoke.log`（路径示例） | **DONE** | RTX 4090；val MPJPE **87.85 mm** @ 2 epochs。 |
| Local | 0 | v29 hierarchical v2 smoke | `outputs/v29_hierarchical_v2_smoke.log`（路径示例） | **FIXED** | RTX 4090；原配置过重导致 smoke 看起来像 hung，已改用轻量脚本 `scripts/run_v29_hierarchical_true_gt_v2_smoke_local_4090_fixed.sh`；2 epochs val MPJPE **95.20 mm**。 |
| Local | 0 | v21 neural BA v2 smoke | `outputs/v21_neural_ba_v2_smoke.log`（路径示例） | **FIXED** | RTX 4090；根因是 `motionflow_mv/fusion/neural_bundle_adjustment_v21.py` 中轴角旋转描述子在单位阵处导数发散产生 NaN，已替换为 `R - R^T` 的反对称部分；修复后 2 epochs val MPJPE **79.42 mm**（从初始 93.50 mm 下降）。

### 已结束但需留意的任务

| 任务 | 状态 | 关键产出 |
|------|------|----------|
| v25 true-GT v2 medium training | ✅ DONE | `outputs/ablations/v25_true_gt_v2_medium_a800.{pth,log}`；best val 31.41 mm |
| v85 random view dropout 训练 | ✅ DONE | `outputs/ablations/v85_random_view_dropout_medium_a800.{pth,log}`；best val 31.42 mm |
| v85 no-fallback 可变视角评估 | ✅ DONE | k=2/3/4 完成，k<4 灾难性 |
| v81/v82/v25 var-view DLT-fallback | ✅ DONE | `outputs/variable_view_fix/variable_view_v{81,82,25}_*_dlt_fallback.{json,csv}` |
| MPI RTMPose 检测 + DLT baseline | ✅ DONE | `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json` |
| AIST++ → H36M 交叉评估 | ✅ DONE | `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json` |
| H36M true-GT v2 审计 | ✅ DONE | DLT 25.67 mm / RANSAC 26.47 mm |

---

## 2. 本地 smoke 结果（RTX 4090 快速验证）

| 任务 | 状态 | 关键结果 | 备注 |
|------|------|----------|------|
| v37 self-critique v2 smoke | **DONE** | val MPJPE **87.85 mm** @ 2 epochs | 已归档。 |
| v21 neural BA v2 smoke | **FIXED** | 修复前初始 **93.50 mm**，修复后 **79.42 mm** @ 2 epochs | 根因：`motionflow_mv/fusion/neural_bundle_adjustment_v21.py` 中轴角旋转描述子在单位阵处导数发散产生 NaN；修复：替换为 `R - R^T` 的反对称部分。 |
| v29 hierarchical v2 smoke | **FIXED** | val MPJPE **95.20 mm** @ 2 epochs | 原配置过重导致看起来像 hung，并非 bug；改用轻量脚本 `scripts/run_v29_hierarchical_true_gt_v2_smoke_local_4090_fixed.sh`。 |
| v39 reliability-coupled graph refinement v2 smoke | **DONE** | val MPJPE **80.52 mm** @ 2 epochs | 基于 v37 模板 + v39 flag；`val_stride=50` 控制单 smoke 约 8 min。 |
| v41 weighted domain loss v2 smoke | **DONE** | val MPJPE **80.23 mm** @ 2 epochs | 基于 v37 模板 + `--domain_loss_weights 1.0,1.5`；`val_stride=50` 控制单 smoke 约 7 min。 |

**说明**：
- v21 修复涉及文件：`motionflow_mv/fusion/neural_bundle_adjustment_v21.py`。轴角旋转描述子在单位阵 `R = I` 处导数发散，优化第一步即产生 NaN；改用 `R - R^T` 的反对称部分后训练稳定。
- v29 使用方法：`CUDA_VISIBLE_DEVICES=0 bash scripts/run_v29_hierarchical_true_gt_v2_smoke_local_4090_fixed.sh`。
- v39 / v41 新 smoke 脚本：`scripts/run_v39_reliability_coupled_graph_refinement_true_gt_v2_smoke_local_4090.sh`、`scripts/run_v41_weighted_domain_loss_true_gt_v2_smoke_local_4090.sh`。
- 本次 smoke 前禁用了 `motionflow_mv/fusion/principal_point_correction.py` 中的大量 `print` 调试输出，否则单次 smoke 会被日志 I/O 拖慢到 30 min 以上。

---

## 3. 已完成的关键里程碑

1. **v85 A800 medium 训练完成**（random view dropout + active-view-count embedding）。
2. **v85 no-fallback 可变视角评估完成**：确认 k<4 仍灾难性，random dropout 本身不足。
3. **h36m_true_gt_v2 数据同步到 A800**：625 MB，manifest 已就绪。
4. **v2 启动脚本同步到 A800**：`scripts/run_v25_true_gt_v2_medium_a800.sh` 与 v86 启动脚本。
5. **A800 磁盘第一轮清理**：腾出约 15 GB，从 99% 降到 98%。
6. **v81/v82/v25 DLT-fallback 可变视角基线完成**，提供 k=2/3/4 的 fallback 参考数字：
   - S9：58.18 / 33.32 / 116.98 mm
   - S11：49.35 / 25.28 / 110.58 mm
7. **MPI 与 AIST++ 数据基线完成**，作为跨域故事锚点。
8. **v21 neural BA v2 smoke 修复并跑通**：根因是 `motionflow_mv/fusion/neural_bundle_adjustment_v21.py` 中轴角旋转描述子在单位阵处导数发散产生 NaN，已替换为 `R - R^T` 的反对称部分；2 epochs val MPJPE 从 **93.50 mm** 降到 **79.42 mm**。
9. **v29 hierarchical v2 smoke 修复并跑通**：原配置过重导致 RTX 4090 上看起来像 hung，并非 bug；改用轻量脚本 `scripts/run_v29_hierarchical_true_gt_v2_smoke_local_4090_fixed.sh` 后 2 epochs val MPJPE **95.20 mm**。

---

## 4. 当前阻塞项与注意事项

### 4.1 GPU 策略违规（最高优先级）

- **GPU 7 被外项目进程占用**：LuxTTS / Mega-ASR / ComfyUI 占用了 GPU 7。
- **处理原则**：
  - **严禁 kill 任何非本项目的进程**。
  - 在 GPU 7 清空或协调好之前，**不得在 GPU 7 启动任何 MotionFlow 任务**。
  - 若 GPU 6 在 v85 DLT-fallback 结束后仍被外项目占用，同样不得启动新任务。
- **可接受动作**：记录进程 PID、占用的 GPU、所属项目，上报给管理员或相关团队协调。

### 4.2 可用 GPU 极度紧张

- 项目只能用 GPU 6/7。
- GPU 6 正在跑 v25 true-GT v2 medium 训练；GPU 7 被外项目占用。
- v85 DLT-fallback 已 kill，v86 排队中；因此 **v85 重跑与 v86 必须在 v25 完成后或 GPU 空闲后才能启动**。

### 4.3 磁盘仍接近满载

- `/mnt/nvme0n1p1` 当前 **98% 满**，剩余约 **73 GB**。
- 任何新的 medium 训练都会继续吃盘，需要：
  - 先跑 `scripts/cleanup_a800_safe.sh` dry-run；
  - 删除已确认废弃的 checkpoint / log（如 v83/v84 失败产物）；
  - 在删除前，确认 v85/v86/v2 结果已经本地或他处备份。

### 4.4 v86 脚本与 v2 数据的对齐

- 已同步的 v86 启动脚本 `scripts/run_v86_no_count_embedding_medium_a800_gpu6.sh` 当前使用的是 `configs/splits/h36m_true_gt_standard.yaml`。
- 若目标是跑在 `h36m_true_gt_v2` 上，**启动前务必确认是否已改为 `configs/splits/h36m_true_gt_v2_standard.yaml`**。
- v25 v2 脚本 `scripts/run_v25_true_gt_v2_medium_a800.sh` 已明确指向 v2 manifest。

### 4.5 本地 git stash 与 token 审计

- **GitHub 清理已完成**：remote URL 中 token 已移除（当前为 `https://github.com/adjcjh777/motionflow-multiview-kimiswarm.git`）、旧工作树 `.worktrees/v18_deformable_attention_baseline` 与本地轻量标签 `v25_local_baseline_monitor_commit` / `v25_local_baseline_monitor_v1` 已删除；`main` 已 push 到 GitHub（commit `d2ed343`）。
- 本地仓库当前仍有 **45 条 stash** 备份在 `patches/stashes/`。
- 部分 stash 可能包含临时写死的 API key / token / 数据库连接串。
- **在 push 或共享任何补丁之前**，先审计 stash：
  - 列出所有 stash 并检查内容；
  - 搜索敏感 token 模式（`sk-...`、`ghp_...`、`AKIA...` 等）；
  - 清理或删除含敏感信息的 stash，切勿直接 push 含 token 的代码。

### 4.6 数据使用禁令

- 继续禁用 `data/h36m_hf/` 和 `data/webbridge/h36m*.npz` 进行模型选择与 leaderboard。
- 所有新 baseline 必须使用 `data/h36m_true_gt_v2/` + `configs/splits/h36m_true_gt_v2_standard.yaml`。

---

## 5. qwen3.8max 的 Next 3 具体任务

> **本地 smoke 状态更新**：
>
> - **v37 self-critique v2 smoke 已完成**：val MPJPE **87.85 mm**（2 epochs），结果已归档。
> - **v29 hierarchical v2 smoke 已修复并跑通**：原配置过重导致 RTX 4090 上看起来像 hung，并非 bug；已改用轻量脚本 `scripts/run_v29_hierarchical_true_gt_v2_smoke_local_4090_fixed.sh`，2 epochs val MPJPE **95.20 mm**。使用方法：`CUDA_VISIBLE_DEVICES=0 bash scripts/run_v29_hierarchical_true_gt_v2_smoke_local_4090_fixed.sh`。
> - **v21 neural BA v2 smoke 已修复并跑通**：根因是 `motionflow_mv/fusion/neural_bundle_adjustment_v21.py` 中轴角旋转描述子在单位阵处导数发散，产生 NaN；修复方式是替换为 `R - R^T` 的反对称部分。修复后 2 epochs val MPJPE 从初始 **93.50 mm** 降到 **79.42 mm**。
>
> **v85 DLT-fallback 评估状态说明**：该任务最初以 PID `2269984` 在 A800 启动，但运行约 29 分钟后进程消失，预期的 JSON/CSV 始终未生成。`outputs/variable_view_fix/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.log` 为空，nohup log 仅显示 `Terminated`（可能是外部 kill 或 OOM）。因此目前没有任何 v85 DLT-fallback 数字；v25/v81/v82 的 DLT-fallback 基线仍是最新参考。该评估需在 GPU 空闲后重跑；重跑前建议检查 eval 脚本是否有输入/输出路径或子进程 hang 的问题。
>
> **v86 A800 看守器**：已部署 `scripts/launch_v86_after_v25_a800.sh`（PID `2337615`），它会等待 v25 true-GT v2 medium 训练完成后在首个可用 GPU 上自动启动 v86 no-count-embedding ablation。v86 不再需要手动 `nohup` 启动；本地 v86 A800 启动看守器已停止。
>
> **v85 post-v86 看守器**：已部署 `scripts/launch_v85_dlt_fallback_after_v86.sh`（PID `2331379`），它会等待 v86 训练完成后在首个可用 GPU 上自动重跑 v85 DLT-fallback 评估。这样在 v86 结束之前不会占用 GPU 6/7 的宝贵训练时间；看守器运行期间只需监控其日志，无需手动启动。
>
> **v21 neural BA camera NaN/Inf 根因确认（已修复）**：失败不是 loader 或 camera 参数问题，而是 `motionflow_mv/fusion/neural_bundle_adjustment_v21.py` 中轴角旋转描述子在单位阵（`R = I`）处导数发散，训练早期即产生 NaN；修复为使用 `R - R^T` 的反对称部分后，本地 RTX 4090 smoke 跑通，2 epochs val MPJPE 从 **93.50 mm** 降到 **79.42 mm**。

### 任务 1：为 v25 true-GT v2 medium 运行 test-set 评估

- v25 true-GT v2 medium 训练已完成：early-stop @ epoch 6，best val MPJPE **31.41 mm**，checkpoint `outputs/ablations/v25_true_gt_v2_medium_a800.pth`。
- 在 GPU 6/7 空闲后，运行 test-set 评估（S9/S11），获得 v2 数据协议下的 test MPJPE/PA-MPJPE。
- 将结果更新到 `docs/results_true_gt_h36m.md` True-GT v2 Leaderboard。
- v25-v2 test 稳定后，再规划 v46/v52/v57/v80/v81/v82 在 v2 上的重跑。

### 任务 2：监控 v86 no-count-embedding ablation 训练

- v86 已启动，tmux session `v86_no_count_embedding`，GPU 6，脚本 `scripts/run_v86_no_count_embedding_medium_a800_gpuX.sh`。
- 监控训练日志：

```bash
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v86_no_count_embedding_medium_a800.log"
```

- **确认**：脚本中的 `--mixed_manifest` 指向 `configs/splits/h36m_true_gt_v2_standard.yaml`。
- v86 目标是验证 active-view-count embedding 对稀疏视角的贡献；与 v85 no-fallback/DLT-fallback 结果对比即可得出结论。

### 任务 3：等待 v85 post-v86 看守器自动重跑 DLT-fallback 评估，并规划 v2 leaderboard 重跑

- **v85 DLT-fallback 将由看守器自动触发**：`scripts/launch_v85_dlt_fallback_after_v86.sh`（PID `2331379`）正在 A800 上运行，等待 v86 训练完成后在首个可用 GPU 启动 v85 DLT-fallback 评估。无需手动 `nohup` 启动。
- 看守器启动后，监控日志：

```bash
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.log"
```

- 评估结束后，对比 v85 DLT-fallback 与 v25/v81/v82 的 DLT-fallback 数字，判断：
  - k=2/3 是否比 no-fallback 有质的改善；
  - k=4 是否有退化；
  - random view dropout 是否在 fallback 模式下有收益。
- 在 v25/v86/v85-DLT-fallback 均完成后，按以下顺序重跑 learned leaderboard：
  - v25、v46、v52、v57、v80、v81、v82 在 v2 数据上重新评估/训练；
  - v85、v86 在 v2 数据上的结果（v85 若此前是 v1 manifest，需补跑 v2 eval；v86 直接跑 v2）。
- 更新 `docs/results_true_gt_h36m.md` 和 `docs/paper_draft_icra_cvpr_2027.md`，确保所有数字都来自 `h36m_true_gt_v2`。

---

## 6. 关键命令速查

### 6.1 查看 v85 DLT-fallback 评估

```bash
# 实时跟踪日志
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.log"

# 查看结果 JSON
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.json"

# 检查进程
ssh a800-D "ps -p 2269984 -o pid,ppid,cmd,%cpu,%mem,etime"
```

### 6.2 监控 v86 A800 看守器与训练日志

```bash
ssh a800-D
# 进入训练仓库
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

# 查看 v86 看守器（等待 v25 完成并自动启动 v86）
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/launch_v86_after_v25_a800.log"

# v86 启动后，查看训练日志
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v86_no_count_embedding_medium_a800.log"

# 检查 v86 看守器进程
ssh a800-D "ps -p 2337615 -o pid,ppid,cmd,%cpu,%mem,etime"

# 确认 GPU 空闲（项目只使用 GPU 6/7）
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv
```

### 6.3 启动 v25 true-GT v2 medium baseline

```bash
ssh a800-D
nohup bash scripts/run_v25_true_gt_v2_medium_a800.sh \
    > outputs/ablations/v25_true_gt_v2_medium_a800.log 2>&1 &
```

### 6.4 检查 GPU 与外项目进程

```bash
# GPU 占用
ssh a800-D "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv"

# 外项目进程（LuxTTS / Mega-ASR / ComfyUI / venv）
ssh a800-D "ps -ef | grep -E 'LuxTTS|Mega-ASR|ComfyUI|venv-cu130-a800' | grep -v grep"

# 查看占用 GPU 7 的具体进程
ssh a800-D "nvidia-smi -i 7 -q -d PIDS"
```

### 6.5 磁盘与清理

```bash
# 磁盘使用情况
ssh a800-D "df -h /mnt/nvme0n1p1"

# 安全清理 dry-run（先不要真删）
ssh a800-D "bash /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/scripts/cleanup_a800_safe.sh --dry-run"

# 查看 A800 outputs 目录大小
ssh a800-D "du -sh /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/* | sort -rh | head -20"
```

### 6.6 本地 git stash / token 审计

```bash
# 统计 stash 数量
git stash list | wc -l

# 列出所有 stash
git stash list

# 搜索常见 token 模式（在本地仓库根目录执行）
grep -R -E "(sk-[a-zA-Z0-9]{48}|ghp_[a-zA-Z0-9]{36,}|AKIA[A-Z0-9]{16}|gh[ops]_[a-zA-Z0-9]{36,})" \
    --include="*.py" --include="*.sh" --include="*.yaml" --include="*.json" . 2>/dev/null

# 查看单个 stash 内容（示例）
git stash show -p stash@{0}
```

### 6.7 查看已有 leaderboard 与基线结果

```bash
# v25 / v82 / v85 no-fallback / DLT-fallback 结果
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.json"
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/variable_view_v82_true_gt_medium_a800_dlt_fallback.json"
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_v85_random_view_dropout_medium_a800.json"

# MPI / AIST++ 基线
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json"
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json"
```

---

## 7. 完成标准 checklist

- [ ] v86 no-count-embedding ablation 训练完成并归档结果。
- [x] v25 true-GT v2 medium baseline 完成训练（best val 31.41 mm @ epoch 6）。
- [ ] v25 true-GT v2 medium test-set 评估完成并更新 leaderboard。
- [ ] v85 DLT-fallback 可变视角评估完成并归档结果（看守器自动触发）。
- [ ] GPU 7 外项目占用问题记录/上报，未 clear 前不强制启动新任务。
- [ ] A800 磁盘继续清理，确保任何新 medium 训练前 ≥5 GB 安全余量。
- [ ] 本地 45 条 stash 完成审计，确认无 token 泄露风险。
- [ ] v2 learned leaderboard（v25/v46/v52/v57/v80/v81/v82/v85/v86）结果更新到 `docs/results_true_gt_h36m.md`。
- [ ] 论文 draft `docs/paper_draft_icra_cvpr_2027.md` 数字与 v2 leaderboard 一致。

---

## 8. 交接注意事项

- **GPU 7 外项目占用**：不要 kill 进程，先记录并协调；未 clear 前不启动任何新任务。
- **v85 DLT-fallback 已排队**：原 PID `2269984` 失败；现在由 `scripts/launch_v85_dlt_fallback_after_v86.sh` 看守器自动触发，等待 v86 完成后在首个可用 GPU 运行，无需手动启动。
- **v25 true-GT v2 medium 已完成**：best val **31.41 mm** @ epoch 6；checkpoint 已落盘；待运行 test-set 评估。
- **v86 已启动**：tmux session `v86_no_count_embedding`，GPU 6，v2 协议；不要手动重复启动。
- **v2 数据是新的标准**：所有新 baseline 必须使用 `data/h36m_true_gt_v2/` + `configs/splits/h36m_true_gt_v2_standard.yaml`。
- **磁盘 98% 满**：启动新的 medium 训练前，先跑 cleanup dry-run 并确认有 ≥5 GB 余量。
- **GitHub 清理已完成**：remote URL token、旧工作树、本地轻量标签已清理，`main` 已 push（commit `d2ed343`）。`patches/stashes/` 中 45 个 stash patch 备份仍保留，审计完成前仍谨慎处理其中的敏感信息。
- **不要动 A800 只读资源**：`/mnt/nvme0n1p1/zhangzy/projects` 与 `motionflow` Docker 仅可查看，禁止写入/重启。
