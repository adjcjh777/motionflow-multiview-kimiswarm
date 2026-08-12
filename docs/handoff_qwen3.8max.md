# MotionFlow-MultiView 接力目标（qwen3.8max）

> **目标**：修复数据地基，建立非循环评估协议，重建可发表水准的排行榜，锚定 CVPR 2027。  
> **当前日期**：2026-08-12 ~12:55 UTC（本次刷新）  
> **仓库**：`D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`  
> **远程只读资源**：A800-D `/mnt/nvme0n1p1/zhangzy/projects` 与 `motionflow` Docker 容器仅供查看，禁止写入或启动训练。  
> **远程可写训练仓库**：`/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`，仅用于在 A800 主机上 `nohup` 启动作业。  
> **GPU 策略**：A800 仅 GPU 6/7 可用于本项目；GPU 0–5 保留，严禁使用。

---

## 1. 核心结论

- **v85 random view dropout 正在 GPU 7 上运行**：PID `2058225`，日志 `outputs/ablations/v85_random_view_dropout_medium_a800.log`。已完成 Epoch 2（train_loss 14.91，val_MPJPE **36.48 mm**），当前 **Epoch 3 进行中**，loss 持续下降（step 1250 约 8.58）。**不要触碰。**
- **v85 无 fallback 可变视角评估正在 GPU 6 上运行**：采用 split-k 模式，当前 **k=2 进行中**；主 launcher PID `2148510`，Python 进程 PID `2148515`。**不要 kill、重启或干扰。** 该 eval 日志输出被缓冲，当前主日志 0 bytes，k2 日志也是 0 bytes，这是预期行为（按 k 批量输出）。
- **GPU 6 约 80 GB 空闲显存**，因此可以接受轻量级只读或短时 smoke 测试，**但不得影响正在运行的 eval**。
- **v85 post-training eval suite monitor**（PID `2072251`）和 **VoxelPose auto-launch monitor**（PID `2146696`）正在等待：eval suite monitor 等 v85 训练结束；VoxelPose monitor 等 eval suite monitor 完成。
- **A800 磁盘仍紧张**：`/mnt/nvme0n1p1` **99% 满**，约 **58 GB** 空闲。
- **v82/v81/v25 可变视角 DLT-fallback 评估已完成**；MPI 检测与 DLT baseline、AIST++ → H36M 交叉评估均已完成。

---

## 2. 正在运行的任务

| 机器 | GPU | 任务 | 日志/输出 | 状态 | 说明 |
|------|-----|------|-----------|------|------|
| A800-D | 7 | v85 random view dropout training | `outputs/ablations/v85_random_view_dropout_medium_a800.log` | **RUNNING** | PID `2058225`；Epoch 2 val_MPJPE **36.48 mm**，Epoch 3 进行中，loss 下降。**禁止 kill/restart。** |
| A800-D | 6 | v85 no-fallback variable-view eval (split-k, k=2 当前) | `outputs/variable_view_v85_random_view_dropout_medium_a800*.log` | **RUNNING** | PID `2148515`（launcher `2148510`）。日志按 k 批量写，当前可能为 0 bytes。**禁止 kill/restart。** |
| A800-D | — | v85 eval suite monitor | `outputs/sota_baselines/monitor_v85_then_run_evals.log` | **RUNNING** | PID `2072251`；训练结束后自动启动 test/no-fallback/DLT-fallback evals |
| A800-D | — | VoxelPose auto-launch monitor | `outputs/sota_baselines/monitor_v85_evalsuite_then_launch_voxelpose.log` | **RUNNING** | PID `2146696`；v85 eval suite 结束后自动启动 VoxelPose |
| A800-D | — | v25/v81/v82 variable-view DLT-fallback | `outputs/variable_view_fix/variable_view_v{25,81,82}_true_gt_*_a800_dlt_fallback.*` | **COMPLETED** | v25 S9 58.18/33.32/116.98 mm；v81 k=2,3 only；v82 k=2/3/4 |
| A800-D | — | MPI RTMPose 检测 + DLT baseline | `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json` | **DONE** | 16/16 `.npz`，DLT baseline 完成 |
| A800-D | — | AIST++ → H36M 交叉评估 | `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json` | **DONE** | combined **~93.94 mm** |
| Local | 0 | — | — | IDLE | RTX 4090 空闲，仅用于 smoke（<30 min） |

---

## 3. 本阶段关键产出

| 任务 | 状态 | 关键产出 |
|------|------|----------|
| v85 random view dropout 训练 | 🔄 RUNNING | GPU 7，PID `2058225`；Epoch 2 val_MPJPE **36.48 mm**；Epoch 3 进行中 |
| v85 no-fallback 可变视角评估 | 🔄 RUNNING | GPU 6，PID `2148515`；当前 k=2；**禁止触碰** |
| v85 eval suite monitor | 🔄 RUNNING | PID `2072251`；训练结束后自动启动 test/no-fallback/DLT-fallback evals |
| VoxelPose auto-launch monitor | 🔄 RUNNING | PID `2146696`；v85 eval suite 结束后自动启动 VoxelPose |
| v82/v81/v25 var-view DLT-fallback | ✅ DONE | `outputs/variable_view_fix/variable_view_v{82,81,25}_true_gt_*_a800_dlt_fallback.*` |
| MPI 16/16 + DLT baseline | ✅ DONE | MPJPE **115.09 mm**，PA-MPJPE **132.68 mm** |
| AIST++ → H36M cross-eval | ✅ DONE | combined **~93.94 mm** |
| H36M true-GT 非循环验证 | ✅ | `data/h36m_true_gt/` direct MJE 0.0000 mm |
| v81/v82 真 GT 主结果 | ✅ | v81 **37.83 mm**，v82 **39.46 mm** |

完整结果见 `docs/results_true_gt_h36m.md`。

---

## 4. 当前真 GT 排行榜（H36M S9/S11）

| 方法 | Combined (mm) | PA-MPJPE (mm) | 备注 |
|---|---:|---:|---|
| Iskakov ICCV 2019 | **23.40** | 23.15 | frozen ref |
| DLT (conf-weighted) | **25.67** | 25.55 | frozen ref |
| RANSAC/conf-DLT | **26.47** | 28.98 | reproducible ref |
| v25 stability | **30.83** | 34.35 | best learned result |
| v25 mixed H36M+AIST++ | 33.42 | 34.60 | early-stopped @ Epoch 3 |
| v81 temporal-pose-attention | 37.83 | 37.75 | completed 8 epochs |
| v82 multi-scale temporal-pose-attention | 39.46 | 39.94 | completed 8 epochs |
| AIST++-only → H36M | 93.94 | 44.50 | zero-shot cross-domain |

---

## 5. 各模块最新状态

### P0 稀疏视角 k<4 学习模型失效

- **v85 训练中**：在 GPU 7 上运行 random view dropout（dropout prob 0.3，min 2 views），配合 active-view-count embedding，让模型在训练时直接见 k=2/3/4。Epoch 2 val_MPJPE **36.48 mm**，Epoch 3 进行中，loss 持续下降。**不要 kill/restart。**
- **v85 no-fallback 可变视角评估**：正在 GPU 6 上以 split-k 模式运行，当前 **k=2**。进程 PID `2148515`，launcher PID `2148510`。日志被有意缓冲到每个 k 完成才写，因此当前主日志和 k2 日志可能为 0 bytes。**禁止 kill/restart。** 如果下一个 agent 接手时发现它已结束，应查看 `outputs/variable_view_v85_random_view_dropout_medium_a800_k{2,3,4}.log` 和最终 `.json/.csv`。
- **DLT-fallback 基线**：v25/v81/v82 可变视角 DLT-fallback 已完成。当前 fallback 数字：S9 k=2/3/4 = 58.18/33.32/116.98 mm；S11 k=2/3/4 = 49.35/25.28/110.58 mm。
- **下一步**：等待 v85 训练结束并自动跑 eval suite；对比 k=2/3/4 与 DLT-fallback 基线；随后 VoxelPose 自动启动。

### P1 MPI-INF-3DHP 真实检测 2D

- **检测**：16/16 `.npz` 已生成。
- **DLT baseline**：mean MPJPE **115.09 mm**，mean PA-MPJPE **132.68 mm**。
- **结论**：RTMPose 检测 2D 与 3D mocap 对齐仍差；如需 learned MPI 结果，先校准 camera/joint 映射；否则把 115.09 mm 作为跨域几何基线。

### P2 AIST++-only 零样本跨域

- **结果**：S9 98.17 mm，S11 89.70 mm，combined ~93.94 mm，PA-MPJPE 44.50 mm。
- **结论**：远高于 60 mm 阈值，**继续暂停 H36M+AIST++ mixed 训练**，集中资源于 v85。

---

## 6. 建议的接力工作清单

1. **等待 v85 完成并评估（最高优先级）**
   - 监控 `outputs/ablations/v85_random_view_dropout_medium_a800.log`（GPU 7，PID `2058225`）。
   - 监控 `outputs/variable_view_v85_random_view_dropout_medium_a800_k{2,3,4}.log` 和 `.json/.csv`（GPU 6，PID `2148515`）。**不要 kill/restart。**
   - eval suite monitor 会自动在训练结束后启动 test/no-fallback/DLT-fallback evals。
   - 对比 k=2/3/4 与 v25 stability / v81 / v82 的 DLT-fallback 数字。

2. **启动 VoxelPose SOTA 基线**
   - VoxelPose 需等待 v85 eval suite 结束后由 monitor（PID `2146696`）自动启动。
   - 确认 A800 上 `models/voxelpose-pytorch/output` 和 `log` 目录已创建。
   - 使用 `CUDA_VISIBLE_DEVICES=6 bash scripts/run_voxelpose_h36m_true_gt_a800.sh` 启动（monitor 会自动处理）。

3. **MPI baseline 归档**
   - 将 `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json` 的关键数字写入 `docs/results_true_gt_h36m.md` 的 MPI 章节。

4. **磁盘清理**
   - A800 `/mnt/nvme0n1p1` 99% 满；v85 运行期间监控空间，必要时运行 `scripts/cleanup_a800_safe.sh` dry-run。

5. **继续 paper rewrite**
   - `docs/paper_draft_icra_cvpr_2027.md` §5.1 与 §5.5.1 已归档 MPI detected-2D DLT baseline 与 AIST++-only cross-eval 最终数字；v85 稀疏视角结果出来后再统一审阅。

---

## 7. 执行约束

- **A800 / Docker 只读**：可 `ssh a800-D` 查看文件，禁止启动/重启 Docker 或 tmux 训练；在 A800 主机训练仓库使用 `nohup` 启动作业是允许的。
- **GPU 状态**：GPU 7 跑 v85 训练（PID `2058225`）；GPU 6 跑 v85 no-fallback eval（PID `2148515`）；GPU 0–3 为 VLLM；GPU 4/5 保留给其他项目，禁止本项目使用。
- **GPU 6 可接受轻量任务**：GPU 6 仍有约 80 GB 空闲显存，因此只读/短时 smoke 测试可以运行，但**必须确保不影响正在运行的 eval**。
- **本地 GPU**：仅用于 smoke/diagnostic（<30 min），当前空闲。
- **不要同时启动多个本地 GPU 训练进程**。
- **数据**：不要使用 `data/h36m_hf/` 或 `data/webbridge/h36m*.npz` 进行模型选择。
- **磁盘**：避免在 A800 上 dump 额外 checkpoint 或抽帧；v85 运行前已 99% 满。

---

## 8. 完成标准

- [x] GPU 策略更新：A800 仅使用 GPU 6/7；GPU 0–5 禁止用于本项目。
- [x] MPI 检测 16/16 完成，DLT baseline 115.09 mm / 132.68 mm。
- [x] AIST++ cross-eval 完成，combined ~93.94 mm。
- [x] v85 已迁移并运行在 GPU 7（PID `2058225`）。
- [x] v82/v81/v25 var-view DLT-fallback 全部完成。
- [ ] v85 训练完成并通过 variable-view eval 验证 k<4 改善。
- [ ] v85 no-fallback split-k eval 完成（GPU 6，当前 k=2 进行中；**禁止 kill/restart**）。
- [ ] VoxelPose SOTA 基线启动并完成（auto-launch monitor PID `2146696` 自动触发）。
- [ ] A800 磁盘清理完成，释放 ≥2 GB。
- [ ] 论文 draft 数字与 `docs/results_true_gt_h36m.md` 一致（待 v85 结果出来后统一审阅）。

---

## 9. 快速入口命令

```bash
# 查看 A800 GPU 状态
ssh a800-D "nvidia-smi"

# 查看 v85 训练日志
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v85_random_view_dropout_medium_a800.log"

# 查看 v85 no-fallback eval（当前 k=2；日志可能按 k 批量写）
ssh a800-D "ls -l /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_v85_random_view_dropout_medium_a800*"

# 查看 v85 post-training eval suite monitor
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/sota_baselines/monitor_v85_then_run_evals.log"

# 查看 v85 VoxelPose auto-launch monitor
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

## 10. 交接注意事项

- **GPU 策略**：本项目仅使用 GPU 6/7，GPU 0–5 禁止。
- **GPU 6 已占用**：v85 no-fallback split-k eval 正在运行（PID `2148515`），当前 k=2。**禁止 kill/restart。** 轻量 smoke 测试可运行，但需保证不影响该 eval。
- **GPU 7 已占用**：v85 training 正在运行（PID `2058225`）。**禁止 kill/restart。**
- **v81/v82 使用 manifest 模式**：`--dataset_manifest tmp/h36m_true_gt_val_manifest.txt`，输出到 `variable_view_fix/variable_view_v{81,82}_true_gt_medium_a800_dlt_fallback.json`。
- **MPI 不要跑 learned model**：DLT baseline 115.09 mm，显著高于 ~20–30 mm 目标，需先检查 camera/joint 对齐。
- **A800 磁盘 99% 满**：当前约 58 GB 空闲，v85 训练期间密切监控，必要时清理。
