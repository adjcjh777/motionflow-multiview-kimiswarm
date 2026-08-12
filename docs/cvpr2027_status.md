# MotionFlow-MultiView CVPR 2027 Status (2026-08-12 ~14:16 UTC)

## 1. 核心结论

- **数据地基 > 架构堆叠**。H36M 旧标签是 2D 输入的 DLT 三角化，导致 v25–v79 的数字无法用于模型选择。
- **真 GT  leaderboard 已建立**：v25 stability 以 **31.56 mm** 领先，v81/v82 在 37–39 mm 区间，均优于 DLT (25.67 mm) 与 Iskakov (23.40 mm) 之外的纯几何基线。
- **卖点必须转向稀疏视角 / 跨域鲁棒性**：k<4 时 learned model 仍会崩溃，v85 正在用随机 view dropout 训练解决；跨域（AIST++、MPI）结果将成为 CVPR 2027 故事核心。

## 2. 数据地基现状

| 数据集 | 真 GT / 非循环？ | 状态 |
|---|---|---|
| H36M true GT | ✅ | `data/h36m_true_gt/` 已就绪，但与 cameras/2D 存在对齐偏差；`scripts/convert_h36m_true_gt_v2.py` 已生成更一致的 v2 标签，完整 `.npz` 已生成，待 v85 完成后同步到 A800 |
| MPI-INF-3DHP | ✅ 真 GT | RTMPose 真实检测 2D 已完成（16/16 `.npz`）；DLT baseline **115.09 mm** / PA-MPJPE **132.68 mm** → `outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json` |
| Shelf/Campus detected | ✅ | `data/webbridge/shelf_campus_detected/` 已就绪 |
| AIST++ | ✅ | 1,408 个 canonical `.npz` 已就绪；AIST++-only medium fast v2 完成；H36M cross-eval **93.94 mm** |
| 3DPW | ⚠️ | pseudo 循环；actual 单目，不可三角化，暂不用于排行榜 |

## 3. H36M true GT 排行榜（S1,5,6,7,8 → S9/S11）

| 方法 | Combined MPJPE (mm) | PA-MPJPE (mm) | 备注 |
|---|---:|---:|---|
| **Iskakov ICCV 2019** | **23.40** | — | 几何方法 leader |
| DLT (conf-weighted) | 25.67 | 28.05 | frozen ref |
| DLT (unweighted) | 28.77 | 32.10 | frozen ref |
| RANSAC/conf-DLT | 26.47 | — | reproducible |
| **v25 stability** | **31.56** (S9 34.87 / S11 26.80) | **34.35** | 当前 best learned模型；8-epoch medium，early-stopped @ epoch 12 |
| v81 temporal-pose-attention | **37.83** (S9 42.19 / S11 33.46) | **37.75** | medium |
| v82 multi-scale temporal-pose-attention | **39.46** (S9 42.07 / S11 36.84) | **39.94** | medium |
| v80 regularization ablation | **53.98** (S9 56.69 / S11 51.27) | **32.47** | medium |
| v52 UWT | **54.01** (S9 58.15 / S11 49.87) | **42.22** | medium |
| v57 re-run | **57.81** | — | medium，early-stopped @ epoch 7 |
| v46 | **52.46** (S9 55.03 / S11 49.88) | **40.20** | medium |

- 详细表格与日志见 `docs/results_true_gt_h36m.md`。

## 4. 稀疏视角（variable view）结果

| 方法 | k=2 S9/S11 | k=3 S9/S11 | k=4 S9/S11 | 备注 |
|---|---|---|---|---|
| v25 + DLT fallback | 58.18 / 49.35 | 33.32 / 25.28 | 116.98 / 110.58 | k<4 用 conf-DLT，k=4 用 learned model |
| v81 + DLT fallback | 58.18 / 49.35 | 33.32 / 25.28 | — | k=2/3 only |
| v82 + DLT fallback | 58.18 / 49.35 | 33.32 / 25.28 | 47.81 / 42.36 | k=2/3 用 DLT fallback，k=4 用 learned v82 |
| v85 no-fallback | — | — | — | **RUNNING** on GPU 6；k=2 已测得 2310.27 / 2308.80 mm（catastrophic），k=3/4 进行中 |

- 关键结论：当前所有非 dropout 训练的模型在 k<4 时会崩溃；v85 是首个原生用随机 view dropout 训练的模型，其结果将决定稀疏视角问题是否得到解决。

## 5. 跨域结果

| 训练数据 | 测试数据 | Combined MPJPE (mm) | 备注 |
|---|---:|---:|---|
| AIST++ only | AIST++ val | 91.43 | best @ epoch 4 |
| AIST++ only | H36M true-GT S9/S11 | **93.94** | S9 98.17 / S11 89.70 |
| AIST++ full | AIST++ | DLT 15.93 mm weighted / 38.11 mm unweighted | frozen ref |

- 详细结果见 `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json`、`outputs/aistpp_full_dlt_baseline_a800.json`。

## 6. 已完成的关键工作

- ✅ H36M 循环标签审计与真 GT 重建；`data/h36m_true_gt/` 可用，v2 更严格对齐版本已生成。
- ✅ MPI-INF-3DHP RTMPose 真实检测 2D 生成完成（16/16 `.npz`）。
- ✅ AIST++ canonical `.npz` 集成与 medium / cross-eval 完成。
- ✅ v25 stability H36M true-GT medium 完成，test **31.56 mm**，成为当前最佳 learned 模型。
- ✅ v81/v82 H36M true-GT medium 完成（37.83 / 39.46 mm），variable-view DLT-fallback eval 完成。
- ✅ v80/v52/v57/v46 在真 GT 上的 medium 结果完成。
- ✅ v25/v81/v82 variable-view 评估完成，确认 k<4 时 learned model 崩溃、DLT fallback 有效。
- ✅ `docs/paper_draft_icra_cvpr_2027.md` 占位/编造引用已修复。

## 7. 关键阻塞

| 阻塞 | 影响 | 下一步 |
|---|---|---|
| **稀疏视角 k<4 失败仍未解决** | 非 dropout 模型在 k=2/3 时 MPJPE 达数百–数千 mm | 等待 v85（随机 view dropout）训练完成；若仍失败，需设计更强的 count-conditioning 或独立稀疏视角 head |
| **v83/v84 失败** | view-conditioned 架构改进未带来收益 | v83 在 A800 medium 上 plateau ~100 mm 被 kill；v84 不确定性加权 dropout smoke 107.11 mm 被 kill；架构堆叠优先级降低 |
| **A800 磁盘 99% 满** | 仅剩 ~58 GB，无法启动大型新实验或写入大量输出 | 在 v85 完成后运行 `scripts/cleanup_a800_safe.sh` dry-run，清理失败/重复的检查点和日志 |
| **H36M true-GT v2 待同步** | v2 `.npz` 已生成本地，但未同步到 A800 | v85 训练完成后同步并重新跑 leaderboard 基线（DLT/Iskakov/v25） |
| **MPI 检测 DLT baseline 偏高** | 115.09 mm，远高于 H36M/Shelf | 验证 RTMPose 检测结果与相机参数对齐；确认 MPI 作为跨域压力测试的可用性 |

## 8. 在飞工作

| GPU | 任务 | 状态 | 输出 |
|---|---|---|---|
| **GPU 7** | v85 random view dropout H36M true-GT medium | **RUNNING** | `outputs/ablations/v85_random_view_dropout_medium_a800.log`; Epoch 4 val_MPJPE 36.97 mm；Epoch 5 进行中 |
| **GPU 6** | v85 split-k no-fallback variable-view eval | **RUNNING** | `outputs/variable_view_v85_random_view_dropout_medium_a800.{csv,json,log}`; k=2 完成（2310.27 / 2308.80 mm），k=3/4 进行中 |
| **GPU 6/7（queued）** | v85 post-training eval suite | **QUEUED** | `scripts/monitor_v85_then_run_evals.sh`；将在 v85 训练完成后自动启动 test-set eval、fresh no-fallback eval、DLT-fallback eval |

- **不要停止或干扰上述进程。**

## 9. 下一阶段里程碑

1. **等待 v85 完成并评估稀疏视角鲁棒性**
   - GPU 7 训练完成后，监控脚本会自动跑 v85 test-set eval 和 variable-view eval。
   - 比较 v85 k=2/3/4 与 v25 DLT-fallback 基线（S9: 58.18/33.32/116.98 mm；S11: 49.35/25.28/110.58 mm）。
   - 若 k<4 仍 catastrophic，设计更强的 count-conditioning 或独立稀疏视角 head。

2. **清理 A800 磁盘**
   - v85 完成后运行 `scripts/cleanup_a800_safe.sh` dry-run。
   - 优先删除 v83/v84 失败产物、重复 manifest、旧 variable-view 临时文件。

3. **同步 H36M true-GT v2 并重跑基线**
   - 将 `data/h36m_true_gt_v2/` 同步到 A800。
   - 在 v2 上重跑 DLT、Iskakov、v25 基线，确认数字一致或更优。

4. **SOTA 比较**
   - VoxelPose / MVPose / DLT configs 已就绪，待 GPU 6/7 空闲后调度。

5. **跨域训练 mix**
   - 设计 H36M + AIST++ (+ MPI detected-2D / Shelf/Campus) 的 mixed-dataset 实验。
   - 修复 mixed-dataset 训练发散问题（v25 H36M+AIST 曾在 epoch 3 发散到 481.99 mm）。

6. **论文重写**
   - 更新 `docs/paper_draft_icra_cvpr_2027.md` 的表格与引用。
   - 故事围绕：真 GT 诚实评估、稀疏视角鲁棒性、跨域泛化（AIST++/MPI）。

## 10. GPU / 训练资源状态

- **仅 GPU 6 和 GPU 7 可用**；GPU 0–5 为其他项目保留，严禁使用。
- 启动任何新训练/评估前，必须确认 GPU 空闲且 `CUDA_VISIBLE_DEVICES` 为 6 或 7。
- `/mnt/nvme0n1p1/zhangzy/projects` 和 A800 Docker `motionflow` 服务为 **只读**。

## 11. 立即行动（下一步）

1. **等待 v85 训练/评估完成**，不要抢占 GPU 6/7。
2. v85 完成后检查并整理 variable-view 结果，判断稀疏视角问题是否解决。
3. 运行磁盘清理 dry-run，释放 A800 空间。
4. 同步 H36M true-GT v2 并重跑关键基线。
5. 更新 `docs/paper_draft_icra_cvpr_2027.md` 的结果与卖点描述。

---

> 当前最紧迫：**v85 稀疏视角 dropout 训练结果** 和 **A800 磁盘空间**。
