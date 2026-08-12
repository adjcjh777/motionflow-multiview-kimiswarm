# MotionFlow-MultiView CVPR 2027 Status (2026-08-12)

## 1. 核心结论（fable5 修复方向）

- **数据地基 > 架构堆叠**。H36M 旧标签是 2D 输入的 DLT 三角化，导致所有 smoke 数字无法用于模型选择。
- **修复后数字会落在 15–30 mm**，与多视角 H36M SOTA 对齐。
- **卖点必须转向稀疏视角 / 跨域鲁棒性**，而非绝对 MPJPE 碾压。

## 2. 数据地基现状

| 数据集 | 真 GT / 非循环？ | 状态 |
|---|---|---|
| H36M true GT | ✅ | `data/h36m_true_gt/` 已就绪；标准协议 S1,5,6,7,8→S9/S11 manifest 已创建 |
| MPI-INF-3DHP | 真 GT，但 2D 是 GT 投影 | 真实检测 2D 已生成（16 个 `_m.npz` 文件，含合并后的 `s_01_seq_01_02`）于 `data/webbridge/mpi_inf_3dhp_detected_2d/`；`s_02_seq_02` 因相机/标签对齐问题已移除；当前 DLT baseline 在该集合上仍达 ~326–400 mm，存在相机/坐标系对齐问题 |
| Shelf/Campus detected | ✅ | `data/webbridge/shelf_campus_detected/` 已生成 |
| AIST++ | ✅ | 非循环，正在集成 |
| 3DPW | ⚠️ | pseudo 循环；actual 单目，不可三角化 |

## 3. 排行榜

### H36M true GT (S1,5,6,7,8 → S9/S11)

| 方法 | Combined direct (mm) | Combined PA-MPJPE (mm) | 备注 |
|---|---:|---:|---|
| **Iskakov ICCV 2019** | **23.35** | **23.10** | 当前 leader |
| DLT (conf-weighted) | 25.67 | 28.05 | frozen ref |
| DLT (unweighted) | 28.77 | 32.10 | frozen ref |
| v80 (medium) | **39.98** | — | local 8-epoch medium; best epoch 4; overfit to 133.71 mm by epoch 8 |
| v80 (best converged, v3) | **42.60** | — | local 2 epochs; A800 v2 best 39.70 |
| v80 (smoke) | 98.12 | — | 2-epoch smoke |
| **v25** | **43.93** (test) | — | 8-epoch medium; corrected-val ablations 45.80 / 46.75 mm @ epoch 1; diverged |
| **v57** | **57.81** (re-run best) / **80.21** (old final) / **75.16** (old obs.) | — | re-run finished, early-stopped @ epoch 7; old 5-epoch medium final 80.21 mm |

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
- ✅ v57 H36M true-GT medium 训练完成（5 epoch，最佳 val MPJPE 75.16 mm（观测值）/ 81.47 mm（保存的 ckpt），epoch 3；最终 80.21 mm）。
- ✅ **v25 true-GT divergence ablations 完成**（`v25_true_gt_baseline_fix` 在 A800 GPU 4 上最佳 45.80 mm @ epoch 1，最终发散至 323.35 mm；`v25_true_gt_geometry_regularization_a800` 在 A800 GPU 6 上最佳 46.75 mm @ epoch 1，最终发散至 281.22 mm）。GPU 4/6 已释放。
- ✅ v80 H36M true-GT medium 训练完成（8 epoch，最佳 val MPJPE 39.98 mm，epoch 4；后续过拟合至 133.71 mm）。
- ✅ **v57 true-GT re-run 已完成**（best val 57.81 mm @ epoch 4，early-stopped @ epoch 7；checkpoint monitor 已修复为 `mpjpe`）。
- 🔄 **MPI-INF-3DHP RTMPose 真实检测 2D 在 A800 GPU 7 上重新生成中**（duplicate process 已移除）；旧 MediaPipe 检测的 DLT baseline 仍达 ~326–400 mm。
- ✅ **AIST++ manifest 已同步到 A800**，可启动完整 medium / cross-domain 训练。

## 5. 关键阻塞

| 阻塞 | 影响 | 下一步 |
|---|---|---|
| MPI-INF-3DHP 真实检测 2D 对齐问题 | RTMPose 重新生成中；旧 MediaPipe 检测 DLT baseline 仍达 ~326–400 mm | 验证 RTMPose 检测结果；在 DLT baseline 降至 ~20–30 mm 前不宜进行 learned-model 排行榜 |
| v25/v80/v57 已跑满但严重过拟合 | 无法判断复杂模型价值 | 分析 v57 re-run 与 v25 ablations；下一步尝试 mixed-dataset 训练或更强正则化 |
| AIST++ 集成 | smoke 已完成，待跑满 epoch/medium | 补充 AIST++ medium 结果，加入跨域训练 mix |

## 6. 下一阶段计划（CVPR 2027）

1. **完成 H36M 真 GT 完整排行榜**：v25 corrected-val ablations（45.80 / 46.75 mm @ epoch 1）、v80 medium（39.98 mm）和 v57 re-run（57.81 mm @ epoch 4，持续中）已更新；统一表格；补充 AIST++ medium 结果。
2. **MPI 真实检测 2D 对齐**：RTMPose 重新生成中；旧 MediaPipe 检测 DLT baseline 仍达 ~326–400 mm。需先验证 RTMPose 结果，再建立 MPI 标准协议排行榜。
3. **跨域训练 mix**：H36M + AIST++ + Shelf/Campus，可能的话加入 MPI detected-2D。AIST++ manifest 已同步到 A800。
4. **消融与鲁棒性曲线**：使用 `eval_variable_views.py` 生成 `MPJPE@k` 曲线。
5. **重写论文**：更新表格、引用、卖点；强调稀疏视角 / 跨域鲁棒性。
6. **MPI 官方服务器提交**：获取官方 test-set 数字。

## 7. GPU / 训练资源状态

- v25 true-GT divergence ablations 已完成（A800 GPU 4/6 已释放）。v57 true-GT re-run 已完成（best 57.81 mm @ epoch 4，early-stopped @ epoch 7）。MPI RTMPose detection 在 A800 GPU 7 上运行中。
- v80 H36M true-GT medium 已结束（最佳 val MPJPE 39.98 mm）。v57 H36M true-GT medium 旧运行已完成（最佳观测 75.16 mm，保存 ckpt 81.47 mm，epoch 3）。
- A800 tmux 训练仍保持停止；A800-D 与 Docker `motionflow` 服务为只读。
- 启动新的 GPU 训练任务前，务必先用 `nvidia-smi` 确认 GPU 空闲且只有一个训练进程运行。

## 8. 立即行动（下一步）

1. 分析 v57 re-run 结果（best 57.81 mm @ epoch 4）与 v25 ablations（45.80 / 46.75 mm @ epoch 1 后发散），决定是否启动 v80 regularisation 或 mixed-dataset 训练。
2. v57 re-run 已完成，GPU 5 已释放；可安排下一项 A800 训练任务。
3. 验证 MPI RTMPose 检测结果；在 DLT baseline 降至 ~20–30 mm 前，暂缓 learned-model 排行榜。
4. 根据真 GT leaderboard 继续更新 `docs/paper_draft_icra_cvpr_2027.md` 的 results 部分。

---

> 当前最紧迫：**v25/v80/v57 在真 GT 上的 fair comparison** 和 **MPI 真实检测 2D 数据获取**。
