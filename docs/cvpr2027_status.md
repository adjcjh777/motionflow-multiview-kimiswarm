# MotionFlow-MultiView CVPR 2027 Status (2026-08-11)

## 1. 核心结论（fable5 修复方向）

- **数据地基 > 架构堆叠**。H36M 旧标签是 2D 输入的 DLT 三角化，导致所有 smoke 数字无法用于模型选择。
- **修复后数字会落在 15–30 mm**，与多视角 H36M SOTA 对齐。
- **卖点必须转向稀疏视角 / 跨域鲁棒性**，而非绝对 MPJPE 碾压。

## 2. 数据地基现状

| 数据集 | 真 GT / 非循环？ | 状态 |
|---|---|---|
| H36M true GT | ✅ | `data/h36m_true_gt/` 已就绪；标准协议 S1,5,6,7,8→S9/S11 manifest 已创建 |
| MPI-INF-3DHP | 真 GT，但 2D 是 GT 投影 | 真实检测 2D 已生成（15/16 序列）于 `data/webbridge/mpi_inf_3dhp_detected_2d/`；仍缺 S2/Seq2 canonical `.npz` |
| Shelf/Campus detected | ✅ | `data/webbridge/shelf_campus_detected/` 已生成 |
| AIST++ | ✅ | 非循环，正在集成 |
| 3DPW | ⚠️ | pseudo 循环；actual 单目，不可三角化 |

## 3. 排行榜

### H36M true GT (S1,5,6,7,8 → S9/S11)

| 方法 | Combined direct (mm) | Combined PA-MPJPE (mm) | 备注 |
|---|---:|---:|---|
| DLT (unweighted) | 29.19 | 29.31 | frozen ref |
| DLT (conf-weighted) | 25.87 | 25.55 | frozen ref |
| **Iskakov ICCV 2019** | **23.35** | **23.10** | 当前 leader |
| v80 (medium) | **39.98** | — | local 8-epoch medium; best epoch 4; overfit to 133.71 mm by epoch 8 |
| v80 (best converged, v3) | **42.60** | — | local 2 epochs; A800 v2 best 39.70 |
| v80 (smoke) | 98.12 | — | 2-epoch smoke |
| **v25** | **72.80** | — | 8-epoch medium; best epoch 2; S9 67.92 / S11 77.68; overfit to 207.62 mm by epoch 8 |

- 完整结果：`docs/results_true_gt_h36m.md`

### AIST++ smoke （新集成）

| 方法 | val MPJPE (mm) | 备注 |
|---|---:|---|
| DLT (unweighted) | **12.66** | frozen ref |
| DLT (conf-weighted) | **6.52** | frozen ref |
| Iskakov ICCV 2019 | **9.31** | CPU smoke, best epoch 6 |
| v25 | 71.79 | 3-epoch smoke |
| v80 | 76.34 | 3-epoch smoke |

- 说明：AIST++ 真 GT 已接入，以上为首次 3-epoch smoke 结果；v25 略优于 v80，两者均显著低于 DLT 基线，需进一步跑满 epoch 验证收敛性。

### Shelf/Campus detected

| 方法 | Val direct (mm) | 备注 |
|---|---:|---|
| Iskakov | 128.73 | leader |
| DLT (conf-weighted) | 132.29 | — |
| v80 | 408.58 | 3-epoch smoke |
| v57 | 424.63 | 3-epoch smoke |
| v25 | 430.67 | 3-epoch smoke |

- 完整结果：`docs/results_true_gt_shelf_campus.md`

## 4. 已完成的关键工作

- ✅ 停止 A800 训练，确认无训练进程。
- ✅ 闭环验证 H36M 循环标签；生成/确认 H36M true GT。
- ✅ 修复 mixed loader 的 domain embedding 越界。
- ✅ 生成非循环 Shelf/Campus detected `.npz`。
- ✅ 复现 Iskakov ICCV 2019 基线，在 H36M 和 Shelf/Campus 真 GT 上取得可靠数字。
- ✅ 15-agent 数据审计，确认各数据集循环性。
- ✅ 修复 `docs/paper_draft_icra_cvpr_2027.md` 中的占位/编造引用。
- ✅ 创建 CVPR 2027 状态文档（本文件）。
- ✅ AIST++ 数据接入与 smoke 评测（v25/v80/DLT 基线）。
- ✅ v25 H36M true-GT medium 训练完成（8 epoch，最佳 val MPJPE 72.80 mm，epoch 2；后续过拟合至 207.62 mm）。
- ✅ MPI-INF-3DHP 真实检测 2D 生成完成 15/16 序列（`data/webbridge/mpi_inf_3dhp_detected_2d/`），仅缺 S2/Seq2 canonical `.npz`。

## 5. 关键阻塞

| 阻塞 | 影响 | 下一步 |
|---|---|---|
| MPI-INF-3DHP S2/Seq2 canonical `.npz` 缺失 | 训练集缺少 1/16 序列，检测 2D 跳过该序列 | 重新生成/定位 S2/Seq2 canonical `.npz`；补齐 `data/webbridge/mpi_inf_3dhp_detected_2d/s_02_seq_02_v14_multiview_m.npz` |
| v25/v80 已跑满但严重过拟合；v57 仍缺 medium 结果 | 无法判断复杂模型价值 | 补跑 v57 的 8–10 epoch medium，并尝试早停 / 正则化 / SWA 抑制过拟合 |
| AIST++ 集成 | smoke 已完成，待跑满 epoch/medium | 补充 AIST++ medium 结果，加入跨域训练 mix |

## 6. 下一阶段计划（CVPR 2027）

1. **完成 H36M 真 GT 完整排行榜**：v25 medium（72.80 mm）和 v80 medium（39.98 mm）已完；补充 v57 medium，统一表格；同时补充 AIST++ medium 结果。
2. **MPI 真实检测 2D**：已用 MediaPipe 在 `data/webbridge/mpi_inf_3dhp_detected_2d/` 生成 15/16 序列；补齐缺失的 S2/Seq2 canonical `.npz` 后重跑该序列即可。
3. **跨域训练 mix**：H36M + AIST++ + Shelf/Campus，可能的话加入 MPI detected-2D。
4. **消融与鲁棒性曲线**：使用 `eval_variable_views.py` 生成 `MPJPE@k` 曲线。
5. **重写论文**：更新表格、引用、卖点；强调稀疏视角 / 跨域鲁棒性。
6. **MPI 官方服务器提交**：获取官方 test-set 数字。

## 7. GPU / 训练资源状态

- 截至当前检查，`nvidia-smi` 显示 RTX 4090 利用率 ~22–35%，内存占用 ~2.0 GB，无 `python.exe` GPU 训练进程。GPU 当前**空闲**。
- `agent-51`（v25 H36M true-GT medium）已结束；v80 H36M true-GT medium 也已结束。
- A800 tmux 训练仍保持停止；A800-D 与 Docker `motionflow` 服务为只读。
- 启动新的 GPU 训练任务前，务必先用 `nvidia-smi` 确认 GPU 空闲且只有一个训练进程运行。

## 8. 立即行动（下一步）

1. 复测 v25 最佳 checkpoint（epoch 2）在 S9/S11 上的 per-split 指标与 EMA  shadow 权重。
2. 在 H36M 真 GT 上补跑 v57 medium。
3. 补齐 MPI-INF-3DHP S2/Seq2 canonical `.npz` 并生成对应的真实检测 2D。
4. 根据真 GT leaderboard 重写 `docs/paper_draft_icra_cvpr_2027.md` 的 results 部分。

---

> 当前最紧迫：**v25/v80/v57 在真 GT 上的 fair comparison** 和 **MPI 真实检测 2D 数据获取**。
