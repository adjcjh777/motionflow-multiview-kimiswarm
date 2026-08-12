# MotionFlow-MultiView 接力目标（qwen3.8max）

> **目标**：修复数据地基，建立非循环评估协议，重建可发表水准的排行榜，锚定 CVPR 2027。  
> **当前日期**：2026-08-12 ~15:12 UTC（本次刷新）  
> **仓库**：`D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`  
> **远程只读资源**：A800-D `/mnt/nvme0n1p1/zhangzy/projects` 与 `motionflow` Docker 容器仅供查看，禁止写入或启动训练。  
> **远程可写训练仓库**：`/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`，仅用于在 A800 主机上 `nohup` 启动作业。  
> **GPU 策略**：A800 仅 GPU 6/7 可用于本项目；GPU 0–5 保留，严禁使用。

---

## 1. 核心结论

- **v85 训练已完成**：A800 GPU 7 上的 random view dropout 训练已结束；v85 无 fallback 可变视角评估也已在 GPU 6 上完成。
- **v85 无 fallback 可变视角结果（split-k）：**
  - k=2：S9 **2310.27 mm**，S11 **2308.80 mm**
  - k=3：S9 **1119.45 mm**，S11 **1118.18 mm**
  - k=4：S9 **83.52 mm**，S11 **77.07 mm**
  - **k<4 仍然灾难性**；仅靠 random view dropout 无法解决稀疏视角问题。
- **GPU 策略违规**：GPU 6/7 上目前存在其他项目进程（LuxTTS、Mega-ASR、`.venv-cu130-a800`），违反本项目仅使用 GPU 6/7 的约定。**不要 kill 这些进程**，但需记录并上报该违规。
- **v86 状态不确定**：当前进程列表中不可见，需确认其是否完成、崩溃或被覆盖。
- **A800 磁盘仍紧张**：`/mnt/nvme0n1p1` **99% 满**，约 **58 GB** 空闲。
- **下一步**：等待/运行 v85 DLT-fallback 可变视角评估、A800 清理、同步 v2 标签、重跑 leaderboard。

---

## 2. 正在运行的任务

| 机器 | GPU | 任务 | 日志/输出 | 状态 | 说明 |
|------|-----|------|-----------|------|------|
| A800-D | 6/7（queued） | v85 post-training eval suite monitor | `outputs/sota_baselines/monitor_v85_then_run_evals.log` | **RUNNING** | PID `2218949`；训练结束后自动启动 test/no-fallback/DLT-fallback evals。 |
| A800-D | 7 | v85 random view dropout training | `outputs/ablations/v85_random_view_dropout_medium_a800.log` | **DONE** | PID `2058225`；训练已完成。 |
| A800-D | 6 | v86 no-count-embedding ablation | `outputs/ablations/v86_no_count_embedding_medium_a800.log` | **UNCERTAIN** | PID `2203020`；当前进程列表不可见，需确认状态。 |
| A800-D | 6/7 | 其他项目进程 | — | **OCCUPIED** | LuxTTS、Mega-ASR、`.venv-cu130-a800` 占用 GPU 6/7；**禁止 kill**，但不得启动新任务。 |
| A800-D | — | v25/v81/v82 variable-view DLT-fallback | `outputs/variable_view_fix/variable_view_v{25,81,82}_true_gt_*_a800_dlt_fallback.*` | **COMPLETED** | v25 S9 58.18/33.32/116.98 mm；v81 k=2,3；v82 k=2/3/4 |
| A800-D | — | MPI RTMPose 检测 + DLT baseline | `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json` | **DONE** | 16/16 `.npz`，DLT baseline 完成 |
| A800-D | — | AIST++ → H36M 交叉评估 | `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json` | **DONE** | combined **~93.94 mm** |
| Local | 0 | — | — | **IDLE** | RTX 4090 空闲，仅用于 smoke（<30 min） |

---

## 3. 本阶段关键产出

| 任务 | 状态 | 关键产出 |
|------|------|----------|
| v85 random view dropout 训练 | ✅ DONE | GPU 7，PID `2058225`；训练已完成 |
| v85 no-fallback 可变视角评估 | ✅ DONE | GPU 6；k=2/3/4 均完成，k<4 灾难性 |
| v85 post-training eval monitor | 🔄 RUNNING | PID `2218949`；将自动启动 test/DLT-fallback evals |
| v86 no-count-embedding | ❓ UNCERTAIN | 需确认进程/日志状态 |
| v82/v81/v25 var-view DLT-fallback | ✅ DONE | `outputs/variable_view_fix/variable_view_v{82,81,25}_true_gt_*_a800_dlt_fallback.*` |
| MPI 16/16 + DLT baseline | ✅ DONE | MPJPE **115.09 mm**，PA-MPJPE **132.68 mm** |
| AIST++ → H36M cross-eval | ✅ DONE | combined **~93.94 mm** |
| H36M true-GT v2 审计 | ✅ | DLT **25.67 mm**，RANSAC **26.47 mm** |

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
| v85 (no-fallback, k=4) | 83.52 / 77.07 | — | test-set 完整评估待 DLT-fallback 后补充 |

> **注意**：v85 训练已完成，但完整 test-set 评估仍在排队；k=4 no-fallback 结果（S9 83.52 / S11 77.07 mm）弱于 v82，说明 random dropout 损害了全视角性能。

---

## 5. 各模块最新状态

### P0 稀疏视角 k<4 学习模型失效

- **v85 训练已完成**：random view dropout（dropout prob 0.3，min 2 views）+ active-view-count embedding。训练已结束，但 k<4 仍灾难性。
- **v85 no-fallback 可变视角评估**：已完成。
  - k=2：S9 **2310.27 mm**，S11 **2308.80 mm**
  - k=3：S9 **1119.45 mm**，S11 **1118.18 mm**
  - k=4：S9 **83.52 mm**，S11 **77.07 mm**
- **结论**：random view dropout 单独无法解决 k<4 问题，且 k=4 性能下降（对比 v82 S9 47.81 / S11 42.36 mm）。需要更强 count-conditioning 或专用 sparse-view head。
- **DLT-fallback 基线**：v25/v81/v82 已完成；S9 k=2/3/4 = 58.18/33.32/116.98 mm，S11 = 49.35/25.28/110.58 mm。
- **下一步**：等待 v85 DLT-fallback 评估，对比 no-fallback 与 fallback 数字。

### P1 MPI-INF-3DHP 真实检测 2D

- **检测**：16/16 `.npz` 已生成。
- **DLT baseline**：mean MPJPE **115.09 mm**，mean PA-MPJPE **132.68 mm**。
- **结论**：RTMPose 检测 2D 与 3D mocap 对齐仍差；如需 learned MPI 结果，先校准 camera/joint 映射；否则把 115.09 mm 作为跨域几何基线。

### P2 AIST++-only 零样本跨域

- **结果**：S9 98.17 mm，S11 89.70 mm，combined ~93.94 mm，PA-MPJPE 44.50 mm。
- **结论**：远高于 60 mm 阈值，**继续暂停 H36M+AIST++ mixed 训练**，集中资源于 v85 后续分析与稀疏视角改进。

---

## 6. 建议的接力工作清单

1. **等待/确认 v85 DLT-fallback 可变视角评估（最高优先级）**
   - 监控 `outputs/sota_baselines/monitor_v85_then_run_evals.log`（PID `2218949`）。
   - 若 monitor 因 GPU 占用未触发，确认何时有可用 GPU（需先解决 GPU 违规占用问题）。
   - 对比 v85 DLT-fallback k=2/3/4 与 v25/v81/v82 的 DLT-fallback 数字。

2. **A800 磁盘清理**
   - `/mnt/nvme0n1p1` 99% 满；先跑 `scripts/cleanup_a800_safe.sh` dry-run。
   - 在确认 v85/v86 结果已安全备份前，不要删除其 checkpoint/log。

3. **同步 v2 标签并重跑 learned leaderboard**
   - 待 GPU 6/7 空闲/违规解除后，同步 `data/h36m_true_gt_v2/` 到 A800。
   - 重跑 v25、v46、v52、v57、v80、v81、v82、v85、v86 的 true-GT v2 协议评估。
   - 更新 `docs/results_true_gt_h36m.md` 与 `docs/paper_draft_icra_cvpr_2027.md`。

---

## 7. 执行约束

- **A800 / Docker 只读**：可 `ssh a800-D` 查看文件，禁止启动/重启 Docker 或 tmux 训练；在 A800 主机训练仓库使用 `nohup` 启动作业是允许的。
- **GPU 状态**：GPU 6/7 被其他项目进程（LuxTTS、Mega-ASR、`.venv-cu130-a800`）占用，**禁止 kill**，也禁止在此情况下启动新 MotionFlow 作业。
- **GPU 0–3 为 VLLM**，GPU 4/5 保留给其他项目，禁止本项目使用。
- **本地 GPU**：仅用于 smoke/diagnostic（<30 min），当前空闲。
- **不要同时启动多个本地 GPU 训练进程**。
- **数据**：不要使用 `data/h36m_hf/` 或 `data/webbridge/h36m*.npz` 进行模型选择。
- **磁盘**：避免在 A800 上 dump 额外 checkpoint 或抽帧；v85 运行前已 99% 满。

---

## 8. 完成标准

- [x] GPU 策略更新：A800 仅使用 GPU 6/7；GPU 0–5 禁止用于本项目。
- [x] MPI 检测 16/16 完成，DLT baseline 115.09 mm / 132.68 mm。
- [x] AIST++ cross-eval 完成，combined ~93.94 mm。
- [x] v85 训练完成（PID `2058225`）。
- [x] v85 no-fallback split-k 评估完成（k=2/3/4）。
- [ ] v85 DLT-fallback 可变视角评估完成（monitor PID `2218949` 自动触发）。
- [ ] v86 状态确认（完成/崩溃/被覆盖）。
- [ ] GPU 6/7 违规占用问题记录/上报，恢复前不启动新作业。
- [ ] A800 磁盘清理完成，释放 ≥2 GB。
- [ ] 同步 v2 标签并重跑 learned leaderboard。
- [ ] 论文 draft 数字与 `docs/results_true_gt_h36m.md` 一致。

---

## 9. 快速入口命令

```bash
# 查看 A800 GPU 状态
ssh a800-D "nvidia-smi"

# 查看 v85 post-training eval suite monitor
ssh a800-D "tail -f /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/sota_baselines/monitor_v85_then_run_evals.log"

# 查看 v85 训练最终状态
ssh a800-D "tail -n 50 /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v85_random_view_dropout_medium_a800.log"

# 查看 v85 no-fallback 可变视角评估结果
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_v85_random_view_dropout_medium_a800.json"
ssh a800-D "ls -l /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_v85_random_view_dropout_medium_a800*"

# 查看 v86 状态
ssh a800-D "ls -l /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v86_no_count_embedding_medium_a800*"
ssh a800-D "tail -n 20 /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/ablations/v86_no_count_embedding_medium_a800.log"

# 检查 GPU 占用进程/违规情况
ssh a800-D "ps -ef | grep -E 'LuxTTS|Mega-ASR|venv-cu130-a800' | grep -v grep"

# 查看磁盘
ssh a800-D "df -h /mnt/nvme0n1p1"

# 查看 v82/v25 var-view DLT-fallback
ssh a800-D "cat /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/outputs/variable_view_fix/variable_view_v82_true_gt_medium_a800_dlt_fallback.json"
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

- **GPU 策略违规**：GPU 6/7 上有 LuxTTS、Mega-ASR、`.venv-cu130-a800` 等其他项目进程。**禁止 kill**，但不得启动新 MotionFlow 作业，直到违规占用清除或明确协调好。
- **v85 DLT-fallback 评估**：由 PID `2218949` 自动排队触发；需确认 monitor 未被 GPU 占用阻塞。
- **v86 状态**：当前不可见；接手后先检查 log 与 checkpoint，确认是否完成、崩溃或被覆盖。
- **磁盘 99% 满**：约 58 GB 空闲；v85 相关结果未安全备份前不要清理其文件。
- **数据**：继续禁用 `data/h36m_hf/` 与 `data/webbridge/h36m*.npz` 进行模型选择。
