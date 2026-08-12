# MotionFlow-MultiView 接力目标（qwen3.8max）

> **目标**：修复数据地基，建立非循环评估协议，重建可发表水准的排行榜，锚定 CVPR 2027。  
> **当前日期**：2026-08-12 ~11:45 UTC（本次刷新）  
> **仓库**：`D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`  
> **远程只读资源**：A800-D `/mnt/nvme0n1p1/zhangzy/projects` 与 `motionflow` Docker 容器仅供查看，禁止写入或启动训练。  
> **远程可写训练仓库**：`/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`，仅用于在 A800 主机上 `nohup` 启动作业。

---

## 1. 核心结论

- **v85 random view dropout 正在 GPU 7 上运行（已重启）**：PID `2058225`，日志 `outputs/ablations/v85_random_view_dropout_medium_a800.log`；训练脚本 `scripts/run_v85_random_view_dropout_medium_a800_gpu7.sh`。此前因发现多个命令行相同的 DataLoader worker 进程而被判定为重复启动并 kill，随后确认实为 DataLoader workers，训练进程已重启。当前 step ~550，loss 从 ~24 降至 ~20，仍在下降。目标是从根本上修复 k<4 稀疏视角失效。
- **v85 no-fallback 可变视角评估正在 GPU 6 上运行**：PID `2062181`（launcher `2062178`），使用 `PYTHONUNBUFFERED=1` 启动，日志/输出文件当前仍为 0 bytes（仍在加载/缓冲中）。
- **VoxelPose SOTA 基线已排队**：monitor 脚本 `scripts/monitor_v85_then_launch_voxelpose.sh`（PID `2067976`）将在 v85 no-fallback 评估结束且 GPU 6 显存低于 1000 MiB 后自动启动 VoxelPose。环境、数据、patch 均已就绪；`scripts/patch_voxelpose_function.py` 已改为幂等，会先恢复原始 `function.py` 再正确打 patch。
- **v82 / v81 / v25 可变视角 DLT-fallback 评估已完成**：输出分别位于 `outputs/variable_view_fix/variable_view_v{82,81,25}_true_gt_*_a800_dlt_fallback.*`。
- **MPI-INF-3DHP 检测 16/16 完成**，DLT baseline 已自动跑完：mean MPJPE **115.09 mm**，mean PA-MPJPE **132.68 mm**（`outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json`）。
- **AIST++-only → H36M 交叉评估完成**：S9 **98.17 mm**，S11 **89.70 mm**，combined **~93.94 mm**（`outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json`）。
- **A800 磁盘仍紧张**：`/mnt/nvme0n1p1` **99% 满**，约 **46 GB** 空闲（较上次减少）。

---

## 2. 正在运行的任务

| 机器 | GPU | 任务 | 日志/输出 | 状态 | 说明 |
|------|-----|------|-----------|------|------|
| A800-D | 7 | v85 random view dropout training | `outputs/ablations/v85_random_view_dropout_medium_a800.log` | RUNNING | PID `2058225`；已重启，step ~550，loss ~24 → ~20 下降；GPU 0–5 已禁用，本项目仅 GPU 6/7 |
| A800-D | 6 | v85 random view dropout var-view (no DLT fallback) | `outputs/variable_view_v85_random_view_dropout_medium_a800.*` | RUNNING | PID `2062181`（launcher `2062178`）；`PYTHONUNBUFFERED=1`；输出文件当前 0 bytes |
| A800-D | 6 | VoxelPose SOTA baseline monitor | `scripts/monitor_v85_then_launch_voxelpose.sh` | QUEUED | PID `2067976`；v85 eval 结束且 GPU 6 显存 < 1000 MiB 后自动启动 |
| A800-D | — | v25/v81/v82 variable-view DLT-fallback | `outputs/variable_view_fix/variable_view_v{25,81,82}_true_gt_*_a800_dlt_fallback.*` | COMPLETED | v25 S9 58.18/33.32/116.98 mm；v81 k=2,3 only；v82 k=2/3/4 |
| A800-D | 7 | MPI RTMPose 检测 + DLT baseline | `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json` | DONE | 16/16 `.npz`，DLT baseline 完成 |
| Local | 0 | — | — | IDLE | RTX 4090 空闲，仅用于 smoke（<30 min） |

**空闲 GPU**：GPU 0–5 保留给其他项目；GPU 6/7 为本项目专用。GPU 6 当前运行 v25/v81/v82 三个评估，GPU 7 运行 v85。GPU 7 仍有 ~13 GB MPI 残留显存，可释放。

---

## 3. 本阶段关键产出

| 任务 | 状态 | 关键产出 |
|------|------|----------|
| v85 random view dropout 训练 | 🔄 RUNNING | GPU 7，PID `2058225`；已重启，step ~550，loss ~24 → ~20 下降 |
| v85 no-fallback 可变视角评估 | 🔄 RUNNING | GPU 6，PID `2062181`（launcher `2062178`）；`PYTHONUNBUFFERED=1`；输出仍为空 |
| VoxelPose SOTA 基线 | ⏳ QUEUED | PID `2067976`；v85 eval 结束且显存 < 1000 MiB 后自动启动 |
| v82/v81/v25 var-view DLT-fallback | ✅ DONE | `outputs/variable_view_fix/variable_view_v{82,81,25}_true_gt_*_a800_dlt_fallback.*` |
| MPI 16/16 + DLT baseline | ✅ DONE | MPJPE **115.09 mm**，PA-MPJPE **132.68 mm** |
| AIST++ → H36M cross-eval | ✅ DONE | combined **~93.94 mm** |
| H36M true-GT 非循环验证 | ✅ | `data/h36m_true_gt/` direct MJE 0.0000 mm |
| v81/v82 真 GT 主结果 | ✅ | v81 **37.83 mm**，v82 **39.46 mm** |

---

## 4. 当前真 GT 排行榜（H36M S9/S11）

| 方法 | Combined (mm) | PA-MPJPE (mm) | 备注 |
|---|---:|---:|---|
| Iskakov ICCV 2019 | **23.35** | 23.15 | frozen ref |
| DLT (conf-weighted) | **25.67** | 25.55 | frozen ref |
| RANSAC/conf-DLT | **26.47** | 28.98 | reproducible ref |
| v25 stability | **30.83** | 34.35 | best learned result |
| v25 mixed H36M+AIST++ | 33.42 | 34.60 | early-stopped @ Epoch 3 |
| v81 temporal-pose-attention | 37.83 | 37.75 | completed 8 epochs |
| v82 multi-scale temporal-pose-attention | 39.46 | 39.94 | completed 8 epochs |
| AIST++-only → H36M | 93.94 | 44.50 | zero-shot cross-domain |

完整结果见 `docs/results_true_gt_h36m.md`。

---

## 5. 各模块最新状态

### P0 稀疏视角 k<4 学习模型失效

- **v85 训练中**：在 GPU 7 上已重启 random view dropout（dropout prob 0.3，min 2 views），配合 active-view-count embedding，希望让模型在训练时直接见 k=2/3/4。当前 step ~550，loss 从 ~24 降至 ~20，仍在下降。
- **v85 no-fallback 可变视角评估**：在 GPU 6 上运行中（PID `2062181`，launcher `2062178`），`PYTHONUNBUFFERED=1` 启动；输出文件仍为 0 bytes，说明仍在加载/缓冲。
- **DLT-fallback 基线**：v25/v81/v82 可变视角 DLT-fallback 已完成。当前 fallback 数字：S9 k=2/3/4 = 58.18/33.32/116.98 mm；S11 k=2/3/4 = 49.35/25.28/110.58 mm。
- **VoxelPose SOTA 基线**：已排队（PID `2067976`），将在 v85 no-fallback eval 结束且 GPU 6 显存 < 1000 MiB 后自动启动。patch 脚本已改为幂等，原始 `function.py` 会被正确恢复并重新 patch。
- **下一步**：等待 v85 训练和评估完成，对比 k=2/3/4 与 DLT-fallback 基线；随后启动 VoxelPose 训练。

### P1 MPI-INF-3DHP 真实检测 2D

- **检测**：16/16 `.npz` 已生成。
- **DLT baseline**：mean MPJPE **115.09 mm**，mean PA-MPJPE **132.68 mm**。
- **结论**：RTMPose 检测 2D 与 3D mocap 对齐仍差；如需 learned MPI 结果，先校准 camera/joint 映射；否则把 115.09 mm 作为跨域几何基线。

### P2 AIST++-only 零样本跨域

- **结果**：S9 98.17 mm，S11 89.70 mm，combined ~93.94 mm，PA-MPJPE 44.50 mm。
- **结论**：远高于 60 mm 阈值，**继续暂停 H36M+AIST++ mixed 训练**，集中资源于 v85。

---

## 6. 建议的接力工作清单

1. **等待 v85 完成并评估**
   - 监控 `outputs/ablations/v85_random_view_dropout_medium_a800.log`（GPU 7，PID `2058225`）。
   - 监控 `outputs/variable_view_v85_random_view_dropout_medium_a800.log`（GPU 6，PID `2062181`）。
   - 评估完成后带 `--var_view_dlt_fallback` 重跑对比，比较 k=2/3/4 与 v25 stability / v81 / v82。

2. **监控 VoxelPose 自动启动**
   - monitor 脚本 PID `2067976` 会在 v85 no-fallback eval 结束且 GPU 6 显存 < 1000 MiB 后启动 VoxelPose。
   - 确认 `scripts/patch_voxelpose_function.py` 幂等执行：原始 `function.py` 被恢复并正确 patch。

3. **MPI baseline 归档**
   - 将 `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json` 的关键数字写入 `docs/results_true_gt_h36m.md` 的 MPI 章节。

4. **磁盘清理**
   - A800 `/mnt/nvme0n1p1` 99% 满；v85 运行期间监控空间，必要时运行 `scripts/cleanup_a800_safe.sh` dry-run。

5. **继续 paper rewrite**
   - `docs/paper_draft_icra_cvpr_2027.md` §5.1 与 §5.5.1 已归档 MPI detected-2D DLT baseline 与 AIST++-only cross-eval 最终数字；v85 稀疏视角结果出来后再统一审阅。

---

## 7. 执行约束

- **A800 / Docker 只读**：可 `ssh a800-D` 查看文件，禁止启动/重启 Docker 或 tmux 训练；在 A800 主机训练仓库使用 `nohup` 启动作业是允许的。
- **GPU 状态**：GPU 7 跑 v85 训练（PID `2058225`）；GPU 6 跑 v85 no-fallback eval（PID `2062181` / launcher `2062178`）并排队 VoxelPose monitor（PID `2067976`）；GPU 0–3 为 VLLM；GPU 4/5 保留给其他项目，禁止本项目使用。
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
- [ ] VoxelPose SOTA 基线启动并完成（PID `2067976` monitor 自动触发）。
- [ ] A800 磁盘清理完成，释放 ≥2 GB。
- [ ] 论文 draft 数字与 `docs/results_true_gt_h36m.md` 一致（待 v85 结果出来后统一审阅）。

---

## 9. 快速入口命令

```bash
# 查看 A800 GPU 状态
ssh a800-D "nvidia-smi"

# 查看 v85 训练日志
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v85_random_view_dropout_medium_a800.log"

# 查看 v82 var-view DLT-fallback（S9 / S11 per-dataset 日志）
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/tmp/v82_s9_dlt_fallback.log /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/tmp/v82_s11_dlt_fallback.log"

# 查看 v81 S9 var-view DLT-fallback
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/variable_view_v81_true_gt_medium_a800_dlt_fallback_S9.log"

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

- **GPU 策略**：本项目仅使用 GPU 6/7，GPU 0–5 禁止。v85 需从 GPU 4 迁移到 GPU 7；v25/v81 需从 GPU 5 迁移到 GPU 6/7。
- **GPU 6 已占用/可能被占用**：v82 manifest DLT-fallback eval 正在运行。
- **GPU 7 可用**：MPI RTMPose 检测已完成，仍有 ~13 GB 显存占用，可释放后用于 v85。
- **v81/v82 使用 manifest 模式**：`--dataset_manifest tmp/h36m_true_gt_val_manifest.txt`，输出到 `variable_view_fix/variable_view_v{81,82}_true_gt_medium_a800_dlt_fallback.json`。
- **MPI 不要跑 learned model**：DLT baseline 115.09 mm，显著高于 ~20–30 mm 目标，需先检查 camera/joint 对齐。
- **A800 磁盘 99% 满**：当前约 46 GB 空闲，v85 训练期间密切监控，必要时清理。
