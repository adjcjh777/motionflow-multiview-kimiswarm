# MotionFlow-MultiView 接力目标（qwen3.8max — next）

> **目标**：继续推进 CVPR 2027 发表路径，以 v85 稀疏视角评估与 v86 消融为当前焦点。  
> **当前日期**：2026-08-12 ~14:34 UTC（本次刷新）  
> **仓库**：`D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`  
> **远程只读资源**：A800-D `/mnt/nvme0n1p1/zhangzy/projects` 与 `motionflow` Docker 容器仅供查看，禁止写入或启动训练。  
> **远程可写训练仓库**：`/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`，仅用于在 A800 主机上 `nohup` 启动作业。  
> **GPU 策略**：A800 仅 GPU 6/7 可用于本项目；GPU 0–5 保留，严禁使用。

---

## 1. 核心结论

- **v85 random view dropout 训练已完成**：GPU 7 上 early-stopped @ Epoch 6，最佳 val MPJPE **31.42 mm**，模型保存为 `outputs/ablations/v85_random_view_dropout_medium_a800.pth`。
- **v85 无 fallback 可变视角评估正在 GPU 6 上运行**：PID `2218949`，由 monitor `2072251` 自动启动。正在按 k=2/3/4 split 评估 v85 的稀疏视角表现。当前 nvidia-smi 显示 GPU 6 利用率 34%，占用约 1.5 GB 显存。**不要 kill/restart。**
- **v86 no-count-embedding 消融**：在上一轮 AGENTS.md 中记录为已启动（PID `2203020`），但当前进程列表中未观察到；v85 post-training eval suite 已自动接管 GPU 6。如果 v86 确实被取代，则其消融结论需从后续 log 或 checkpoint 中确认。
- **GPU 7 当前空闲**：v85 训练结束后释放，可供后续任务使用。
- **A800 磁盘仍紧张**：`/mnt/nvme0n1p1` 约 **99% 满**，~58 GB 空闲。
- **v82/v81/v25 可变视角 DLT-fallback 评估已完成**；MPI 检测与 DLT baseline、AIST++ → H36M 交叉评估均已完成。
- **数据地基**：`data/h36m_hf/*.npz` 为循环标签，不可用；`data/h36m_true_gt/*.npz` 与相机/2D 不对齐；修正版 `scripts/convert_h36m_true_gt_v2.py` 已准备。v2 `.npz` 全量生成与 A800 同步仍待 v85 评估完成后进行。

---

## 2. 当前任务状态

| 机器 | GPU | 任务 | 日志/输出 | 状态 | 说明 |
|------|-----|------|-----------|------|------|
| A800-D | 6 | v85 no-fallback variable-view eval | `outputs/variable_view_v85_random_view_dropout_medium_a800.{csv,json}` | **RUNNING** | PID `2218949`；自动启动；**禁止 kill/restart** |
| A800-D | — | v85 post-training eval suite monitor | `outputs/sota_baselines/monitor_v85_then_run_evals*.log` | **RUNNING** | PID `2072251`；会依次启动 test、no-fallback、DLT-fallback evals |
| A800-D | — | VoxelPose auto-launch monitor | `outputs/sota_baselines/monitor_v85_evalsuite_then_launch_voxelpose*.log` | **RUNNING** | PID `2146696`；eval suite 结束后自动启动 VoxelPose |
| A800-D | 7 | v85 random view dropout training | `outputs/ablations/v85_random_view_dropout_medium_a800.log` | **COMPLETED** | Early-stopped @ Epoch 6，best val 31.42 mm |
| A800-D | — | v86 no-count-embedding ablation | `outputs/ablations/v86_no_count_embedding_medium_a800.log` | **UNCERTAIN** | 上一轮记录为 RUNNING；当前未在进程列表中观察到 |
| A800-D | — | v25/v81/v82 var-view DLT-fallback | `outputs/variable_view_fix/variable_view_v{25,81,82}_true_gt_*_a800_dlt_fallback.*` | **COMPLETED** | v25 S9 58.18/33.32/116.98 mm；v81 k=2,3；v82 k=2/3/4 |
| A800-D | — | MPI RTMPose 检测 + DLT baseline | `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json` | **DONE** | 16/16 `.npz`，DLT baseline 完成 |
| A800-D | — | AIST++ → H36M 交叉评估 | `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json` | **DONE** | combined **~93.94 mm** |
| Local | 0 | — | — | IDLE | RTX 4090 空闲，仅用于 smoke（<30 min） |

---

## 3. 本阶段已完成交付物

| 任务 | 状态 | 关键产出 |
|------|------|----------|
| v85 random view dropout 训练 | ✅ DONE | GPU 7 完成；best val **31.42 mm**；checkpoint `outputs/ablations/v85_random_view_dropout_medium_a800.pth` |
| v85 no-fallback 可变视角评估 | 🔄 RUNNING | GPU 6，PID `2218949`；结果将写入 `outputs/variable_view_v85_random_view_dropout_medium_a800.{csv,json}` |
| v85 post-training eval suite monitor | 🔄 RUNNING | PID `2072251`；自动调度后续 evals |
| VoxelPose auto-launch monitor | 🔄 RUNNING | PID `2146696`；eval suite 完成后自动启动 VoxelPose |
| v82/v81/v25 var-view DLT-fallback | ✅ DONE | `outputs/variable_view_fix/variable_view_v{82,81,25}_true_gt_*_a800_dlt_fallback.*` |
| MPI 16/16 + DLT baseline | ✅ DONE | MPJPE **115.09 mm**，PA-MPJPE **132.68 mm** |
| AIST++ → H36M cross-eval | ✅ DONE | combined **~93.94 mm** |
| H36M true-GT v2 converter | ✅ READY | `scripts/convert_h36m_true_gt_v2.py` + `scripts/convert_all_h36m_true_gt_v2.sh`；manifest `configs/splits/h36m_true_gt_v2_standard.yaml` |

完整结果见 `docs/results_true_gt_h36m.md`。

---

## 4. 当前真 GT 排行榜（H36M S9/S11）

| 方法 | Combined (mm) | PA-MPJPE (mm) | 备注 |
|---|---:|---:|---|
| Iskakov ICCV 2019 | **23.40** | 23.15 | frozen ref |
| DLT (conf-weighted) | **25.67** | 25.55 | frozen ref |
| RANSAC/conf-DLT | **26.47** | 28.98 | reproducible ref |
| v25 stability | **30.83** | 34.35 | best learned result |
| v81 temporal-pose-attention | 37.83 | 37.75 | completed 8 epochs |
| v82 multi-scale temporal-pose-attention | 39.46 | 39.94 | completed 8 epochs |
| AIST++-only → H36M | 93.94 | 44.50 | zero-shot cross-domain |

v85 训练 best val **31.42 mm**，test 与 variable-view 结果待 post-training eval suite 完成后补充。

---

## 5. 下一任 agent 的三个具体任务

### 1. 等待 v85 post-training eval suite 完成并解读结果

- 监控 `outputs/ablations/v85_random_view_dropout_medium_a800.log` 确认训练结束（已 early-stopped @ Epoch 6）。
- 监控 `outputs/variable_view_v85_random_view_dropout_medium_a800.{csv,json}`（GPU 6，PID `2218949`）。
- 等待 monitor `2072251` 自动启动 test-set eval 和 DLT-fallback variable-view eval。
- 将 v85 k=2/3/4 数字与以下基线对比：
  - v25 DLT-fallback：S9 58.18/33.32/116.98 mm；S11 49.35/25.28/110.58 mm
  - v81/v82 DLT-fallback 数字
- 判断 random view dropout 是否解决了 k<4 学习失效问题。

### 2. 清理 A800 磁盘并启动 VoxelPose SOTA 基线

- `/mnt/nvme0n1p1` 约 99% 满，~58 GB 空闲。
- 在启动 VoxelPose 前，先运行 `scripts/cleanup_a800_safe.sh` dry-run，识别可安全删除的失败/冗余 checkpoint（如 v83/v84 等）。
- VoxelPose monitor（PID `2146696`）会在 v85 eval suite 结束后自动触发；若未触发，可手动使用 GPU 6 或 7 启动：
  ```bash
  CUDA_VISIBLE_DEVICES=6 bash scripts/run_voxelpose_h36m_true_gt_a800.sh
  ```
  或 GPU 7（确认空闲后）。

### 3. 完成 H36M true-GT v2 数据同步并重建排行榜

- `scripts/convert_all_h36m_true_gt_v2.sh` 生成的 v2 `.npz` 需同步到 A800 `data/h36m_true_gt_v2/`。
- 使用 `configs/splits/h36m_true_gt_v2_standard.yaml` 重新跑 DLT、Iskakov、v25/v81/v82/v85 等基线。
- 更新 `docs/results_true_gt_h36m.md`，确保所有数字来自非循环 v2 协议。
- 同步后需验证 direct MJE 是否在合理范围（<50 mm）。

---

## 6. 执行约束

- **A800 / Docker 只读**：可 `ssh a800-D` 查看文件，禁止启动/重启 Docker 或 tmux 训练；在 A800 主机训练仓库使用 `nohup` 启动作业是允许的。
- **GPU 状态**：GPU 6 跑 v85 no-fallback eval（PID `2218949`）；GPU 7 已空闲；GPU 0–3 为 VLLM；GPU 4/5 保留给其他项目，禁止本项目使用。
- **不要触碰运行中的 v85 eval**：PID `2218949` 正在 GPU 6 上运行，禁止 kill/restart。
- **本地 GPU**：仅用于 smoke/diagnostic（<30 min），当前空闲。
- **数据**：不要使用 `data/h36m_hf/` 或 `data/webbridge/h36m*.npz` 进行模型选择；优先使用 `data/h36m_true_gt_v2/`（同步完成后）。
- **磁盘**：避免在 A800 上 dump 额外 checkpoint 或抽帧；启动 VoxelPose 前先评估清理空间。

---

## 7. 完成标准

- [ ] v85 训练完成 ✅
- [ ] v85 no-fallback variable-view eval 完成（GPU 6，PID `2218949` 运行中；**禁止 kill/restart**）
- [ ] v85 post-training eval suite 自动完成（test + DLT-fallback）
- [ ] VoxelPose SOTA 基线启动并完成（auto-launch monitor PID `2146696` 自动触发，或手动在 GPU 6/7 启动）
- [ ] A800 磁盘清理完成，释放 ≥2 GB
- [ ] H36M true-GT v2 `.npz` 同步到 A800 并重建排行榜
- [ ] 论文 draft 数字与 `docs/results_true_gt_h36m.md` 一致（待 v85 结果出来后统一审阅）

---

## 8. 快速入口命令

```bash
# 查看 A800 GPU 状态
ssh a800-D "nvidia-smi"

# 查看 v85 训练日志（已完成）
ssh a800-D "tail -n 30 /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v85_random_view_dropout_medium_a800.log"

# 查看 v85 no-fallback eval 日志/输出
ssh a800-D "ls -l /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_v85_random_view_dropout_medium_a800*"

# 查看 v85 post-training eval suite monitor
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/sota_baselines/monitor_v85_then_run_evals.log"

# 查看 VoxelPose auto-launch monitor
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/sota_baselines/monitor_v85_evalsuite_then_launch_voxelpose.log"

# 查看 v82 var-view DLT-fallback
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/variable_view_v82_true_gt_medium_a800_dlt_fallback.json"

# 查看 v25 DLT-fallback 结果
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.json"

# 查看 MPI DLT baseline
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json"

# 查看 AIST++ cross-eval
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json"

# 本地 GPU 状态
nvidia-smi

# H36M true-GT 排行榜
cat docs/results_true_gt_h36m.md
```

---

## 9. 交接注意事项

- **GPU 策略**：本项目仅使用 GPU 6/7，GPU 0–5 禁止。
- **GPU 6 已占用**：v85 no-fallback split-k eval 正在运行（PID `2218949`），**禁止 kill/restart**。
- **GPU 7 已空闲**：v85 训练结束后可安全用于 VoxelPose 或其他任务。
- **v85 训练已完成**：best val MPJPE **31.42 mm**，checkpoint 已保存。
- **v86 状态需确认**：上一轮记录为 RUNNING，但当前进程列表未观察到；接手后请先确认 `outputs/ablations/v86_no_count_embedding_medium_a800.log` 和进程状态，避免重复启动。
- **MPI 不要跑 learned model**：DLT baseline 115.09 mm，显著高于 ~20–30 mm 目标，需先检查 camera/joint 对齐。
- **A800 磁盘 99% 满**：启动 VoxelPose 或新训练前务必先清理。
