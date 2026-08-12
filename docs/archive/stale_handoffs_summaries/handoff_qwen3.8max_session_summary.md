# qwen3.8max Session Summary

> **Agent**: qwen3.8max (coder subagent)  
> **Date**: 2026-08-11（最终更新）  
> **Repository**: `D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`  
> **Session purpose**: 更新 qwen3.8max 最终接力文档，固化 A800 v25 消融结果、v57 重跑进展、MPI 检测状态及磁盘约束。

---

## 1. Executive Summary

项目仍处于 **真 GT 协议重建** 阶段。H36M 循环标签已修复，所有模型选择必须基于 `data/h36m_true_gt/`。本阶段重点 A800 v25 消融已结束：baseline 与 geometry regularization 均在 Epoch 1 达到最佳，随后发散；说明单纯正则化/几何损失无法解决真 GT 上的过拟合。v57 用修复后的 trainer 在 GPU 5 重跑，Epoch 4/5 已刷新此前丢失的最佳。

**Key status at session end (2026-08-11 final)**:
- A800 v25 ablations finished early-stopped:
  - `v25_true_gt_baseline_fix`: best **45.80 mm** @ Epoch 1，之后发散。
  - `v25_true_gt_geometry_regularization_a800`: best **46.75 mm** @ Epoch 1，之后发散。
  - GPU 4/6 已释放。
- A800 v57 re-run running on GPU 5:
  - Epoch 4 val **57.81 mm**；Epoch 5 **60.72 mm**。
  - 已超过此前丢失的最佳 **75.16 mm**。
- A800 MPI RTMPose detection running on GPU 7；重复进程已移除。
- A800 磁盘 `/mnt/nvme0n1p1` 99% 满（约 46 GB 空闲），项目输出小，但需避免大量 checkpoint/抽帧。
- RTMPose batch-dimension bug 已修复；AIST++ manifest 已同步；排行榜文档已更新修正数字。

---

## 2. GPU / Compute Status

**A800-D**:
- GPU 4: `v25_true_gt_baseline_fix` — 已完成，空闲。
- GPU 5: `v57_true_gt_medium_a800` — RUNNING。
- GPU 6: `v25_true_gt_geometry_regularization_a800` — 已完成，空闲。
- GPU 7: MPI RTMPose detection — RUNNING。

**Local RTX 4090**: IDLE — 仅用于 quick smoke/diagnostics（<30 min）。

---

## 3. Work Completed in This Session

### 3.1 Documentation / Handoff
- 更新主接力文档：`docs/handoff_qwen3.8max.md`。
- 更新本 session summary：`docs/handoff_qwen3.8max_session_summary.md`。
- 更新 `docs/handoff_next_session.md` 反映 A800 最终状态。

### 3.2 A800 Ablations Final Results

| 任务 | GPU | Best val MPJPE | 状态 | 说明 |
|------|-----|---------------|------|------|
| `v25_true_gt_baseline_fix` | 4 | **45.80 mm** @ epoch 1 | 完成 | 之后发散，GPU 已释放 |
| `v25_true_gt_geometry_regularization_a800` | 6 | **46.75 mm** @ epoch 1 | 完成 | 之后发散，GPU 已释放 |
| `v57_true_gt_medium_a800` | 5 | **57.81 mm** @ epoch 4 | 运行中 | Epoch 5 60.72 mm；已刷新旧最佳 |

- 两个 v25 消融在 Epoch 1 后的 ~46 mm 优于旧 val 72.80 mm，确认 `view_mask` 是主要问题；但均无法阻止后续发散。

### 3.3 True-GT Leaderboard（H36M S1/5/6/7/8 → S9/S11）

| Method | MPJPE (mm) | Notes |
|---|---:|---|
| Iskakov ICCV 2019 | **23.35** | Leader |
| Conf-weighted DLT | **25.67** | Reference |
| RANSAC/conf-DLT | **26.47** | Reproducible reference |
| v80 | **39.98** | Best learned baseline so far |
| v25 | **43.93** | Test result；corrected-val ablations 45.80–46.75 mm @ epoch 1 |
| v57 (re-run) | **57.81** @ epoch 4 | In progress；beats prior lost best 75.16 mm |

### 3.4 Recent Bug Fixes / Infrastructure
- **Trainer best-checkpoint selection**: monitors `mpjpe` instead of `loss`。
- **Validation `view_mask`**: correctly passed through during validation。
- **RTMPose detector**: batch-dimension bug fixed in `scripts/generate_mpi_detected_2d.py`。
- **AIST++ manifest**: `configs/splits/h36m_true_gt_aist_mixed_train_val_a800.yaml` 已同步。
- **文档**: `docs/results_true_gt_h36m.md` 等已更新修正数字。

---

## 4. Active Blockers

| ID | Blocker | Impact | Next Step |
|----|---------|--------|-----------|
| B1 | v25/v80/v57 在真 GT 上过拟合 | 无法信任 learned baselines | 本地 smoke 诊断 LR/warmup/view permutation；稳定后 A800 验证 |
| B2 | MPI 真实检测 2D 质量 | DLT 当前 ~326–400 mm；无法 benchmark learned models | 等 GPU 7 RTMPose 生成完成后重跑 DLT，目标 ~20–30 mm |
| B3 | A800 磁盘 99% 满 | 限制 checkpoint/数据 dump | 仅清理项目自有文件；避免大量输出直到系统清理 |

---

## 5. Key Files Produced or Touched

| File | Purpose |
|------|---------|
| `docs/handoff_qwen3.8max.md` | 主接力文档 |
| `docs/handoff_qwen3.8max_session_summary.md` | 本文件 |
| `docs/handoff_next_session.md` | 快速下阶段入口 |
| `docs/results_true_gt_h36m.md` | H36M true-GT 排行榜 |
| `configs/ablations/v80_true_gt_regularization_a800.yaml` | v80 待启动配置 |
| `scripts/run_v80_ablation_true_gt_regularization_a800.sh` | v80 待启动脚本 |
| `scripts/run_v57_true_gt_medium_a800.sh` | v57 重跑脚本 |
| `motionflow_mv/training/trainer_v2.py` | Trainer best-ckpt fix |
| `scripts/generate_mpi_detected_2d.py` | RTMPose batch-dim fix |

---

## 6. Next Recommended Actions（Prioritized）

1. **继续监控 v57 重跑**（GPU 5），记录最终收敛值与 best ckpt。
2. **本地 RTX 4090 smoke 诊断 v25 Epoch-2 发散**：低 LR、长 warmup、关闭 `variable_view_permute`；<30 min。
3. **启动 v80 真 GT regularization 消融**（GPU 4 或 6），使用已准备的脚本/config。
4. **等待 MPI RTMPose 生成完成**后重跑 DLT baseline。
5. **磁盘管理**：仅删除项目自有 outputs/tmp/extracted frames，不碰其他用户数据。

---

## 7. Commands for Next Agent

```bash
# Check A800 GPU / processes
ssh a800-D "nvidia-smi"

# Tail v57 re-run log
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v57_true_gt_medium_a800.log"

# Check disk space
ssh a800-D "df -h /mnt/nvme0n1p1"

# Local GPU status
nvidia-smi

# H36M true-GT leaderboard
cat docs/results_true_gt_h36m.md
```

---

## 8. Constraints Reminder

- **A800-D / Docker `motionflow`**: READ-ONLY for `/mnt/nvme0n1p1/zhangzy/projects` 与 Docker。
- **A800 训练仓库**: `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20` 可写，启动训练时使用 `CUDA_VISIBLE_DEVICES=X nohup ...`。
- **Local RTX 4090**: 一次仅一个训练任务；当前空闲，仅用于 smoke。
- **不要使用循环 H36M 数据**（`data/h36m_hf/`、`data/webbridge/h36m*.npz`）进行模型选择。
- **磁盘 99% 满**：避免 dump 额外 checkpoint 或抽帧。

---

*End of session summary. Final update 2026-08-11 by qwen3.8max.*
