# MotionFlow-MultiView 接力目标（qwen3.8max）

> **目标**：修复数据地基，建立非循环评估协议，重建可发表水准的排行榜，锚定 CVPR 2027。  
> **当前日期**：2026-08-11  
> **仓库**：`D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`  
> **远程只读资源**：A800-D `/mnt/nvme0n1/zhangzy/projects` 与 `motionflow` Docker 容器仅供查看，禁止写入或启动训练。

---

## 1. 核心结论（2026-08-11）

本项目已从“循环标签陷阱”中恢复，进入**真 GT 基准重建阶段**。

- **H36M 真 GT 已就位**：`data/h36m_true_gt/` 包含标准协议（S1,5,6,7,8 → S9/S11）的真 mocap 世界坐标 `.npz`，通过 reprojection / circularity 双重验收。
- **旧数据仍不可信**：`data/h36m_hf/*.npz`、`data/webbridge/h36m*.npz`、`data/webbridge/h36m_meters/*.npz` 仍为循环标签（direct MJE ≈ 0 mm）。**v25–v79 在这些数据上的所有数字均不能用于模型选择**。
- **真 GT 排行榜已启动**：Iskakov ICCV 2019 在 H36M 真 GT 上达到 **23.35 mm** combined direct，显著优于 DLT。
- **v25 在真 GT 上严重发散**：v25 medium 已完成。best epoch 2 仅 **72.80 mm**，随后过拟合/发散至 epoch 8 的 **207.62 mm**。这是当前最核心的阻塞问题，必须在新一轮模型迭代前定位根因。
- **v80 在真 GT 上同样大幅落后基线**：local medium 8 epoch 最佳 **39.98 mm**（epoch 4），随后过拟合至 epoch 8 的 **133.71 mm**；A800 v2 最佳为 **39.70 mm**。仍显著落后于 Iskakov（23.35 mm）与 confidence-weighted DLT（25.87 mm）。
- **v57 medium 状态未确认**：`nvidia-smi` 未观察到活跃 GPU 训练进程，也未找到 v57 H36M true-GT medium 输出日志；可能尚未启动、已结束或异常退出。
- **卖点已转向**：从“绝对 MPJPE 碾压”转向 **稀疏视角 / 跨域鲁棒性**。
- **GPU 当前状态**：本地 RTX 4090 **空闲**（`nvidia-smi` 显示利用率 ~22–35%，内存 ~2.0 GB，无 `python.exe` GPU 训练进程）。启动新 GPU 任务前请先确认 `nvidia-smi` 仍保持空闲。A800 训练队列仍停止。

---

## 2. 本阶段已完成（2026-08-10 → 2026-08-11）

| 任务 | 状态 | 关键产出 |
|------|------|----------|
| 停止 A800 训练队列 | ✅ | 已确认 A800-D 上无训练进程 |
| 闭环验证 H36M 循环标签 | ✅ | `scripts/diagnose_circular_labels.py` direct MJE ≈ 0 mm |
| 获取并验证 H36M 真 GT | ✅ | `data/h36m_true_gt/` 通过双验收门；S9/S11 DLT error ~25–34 mm |
| 修复混合 loader 的 domain embedding 越界 | ✅ | `experiments/train_omniview_fusion_v5_webbridge_multi.py` 自动根据 manifest 最大 dataset id 设置 `num_domains` |
| 生成非循环 Shelf/Campus .npz | ✅ | `experiments/build_shelf_campus_canonical_from_detection.py` |
| 更新非循环 smoke manifest | ✅ | `configs/splits/shelf_campus_noncircular_smoke.yaml` |
| 复现 Iskakov ICCV 2019 基线 | ✅ | H36M true GT 23.35 mm；Shelf/Campus detected 128.73 mm |
| 15-agent 数据审计 | ✅ | `docs/data_audit_summary_2026-08-11.md` 确认各数据集循环性 |
| AIST++ 非循环验证 | ✅ | DLT error ~44 mm；已创建 `configs/splits/aist_only_smoke.yaml` |
| 修复论文占位/编造引用 | ✅ | `docs/paper_draft_icra_cvpr_2027.md` 已清理 |

### 2.1 关键文件路径

- H36M 真 GT 数据：`data/h36m_true_gt/*_multiview_m.npz`
- H36M 标准协议 manifest：`configs/splits/h36m_true_gt_standard.yaml`
- 诊断脚本：`scripts/diagnose_circular_labels.py`
- true-GT 重投影审计：`scripts/check_true_gt_reprojection.py`
- 转换脚本：`experiments/build_shelf_campus_canonical_from_detection.py`
- H36M true-GT 提取：`experiments/prepare_h36m_true_gt.py` / `scripts/fetch_h36m_true_gt.py`
- Iskakov 基线：`experiments/train_iskakov_baseline_shelf_campus.py`
- 修复 domain embedding 的训练入口：`experiments/train_omniview_fusion_v5_webbridge_multi.py`
- 数据审计摘要：`docs/data_audit_summary_2026-08-11.md`
- CVPR 2027 总览：`docs/cvpr2027_status.md`
- H36M 真 GT 排行榜：`docs/results_true_gt_h36m.md`
- Shelf/Campus 真 GT 排行榜：`docs/results_true_gt_shelf_campus.md`

### 2.2 当前真 GT 排行榜

#### H36M true GT（S1,5,6,7,8 → S9/S11）

| 方法 | Combined direct (mm) | Combined PA-MPJPE (mm) | 备注 |
|---|---:|---:|---|
| DLT (unweighted) | 29.19 | 29.31 | frozen ref |
| DLT (conf-weighted) | 25.87 | 25.55 | frozen ref |
| **Iskakov ICCV 2019** | **23.35** | **23.10** | 当前 leader |
| v80 (medium) | **39.98** | — | best epoch 4; diverged to 133.71 mm by epoch 8 |
| v80 (best converged, v3) | **42.60** | — | local 2 epochs; A800 v2 best 39.70 |
| v80 (smoke) | 98.12 | — | 2-epoch smoke |
| **v25** | **72.80** | — | best epoch 2/8, then divergence to 207.62 |
| v57 (medium) | — | — | status unconfirmed; no active GPU training observed |

- 完整结果：`docs/results_true_gt_h36m.md`
- Iskakov 显著优于 DLT；v80 best converged 42.60 mm 仍落后 DLT/Iskakov；v25 在真 GT 上严重发散，best epoch 2 后持续过拟合。

#### Shelf/Campus detected（真实检测 2D + 真实标注 3D）

| 方法 | Val direct (mm) | 备注 |
|---|---:|---|
| Iskakov ICCV 2019 | **128.73** | leader |
| DLT (conf-weighted) | 132.29 | frozen ref |
| DLT (unweighted) | 134.43 | frozen ref |
| v80 | 408.58 | 3-epoch smoke |
| v57 | 424.63 | 3-epoch smoke |
| v25 | 430.67 | 3-epoch smoke |

- 完整结果：`docs/results_true_gt_shelf_campus.md`
- 所有 learned model 均为 3-epoch smoke，远未收敛。

---

## 3. 剩余关键阻塞（解除后才能继续模型迭代）

### P0-1 H36M 真 3D GT ✅ 已解决

- **状态**：本地 `data/h36m_true_gt/` 已有真 GT `.npz`，通过 reprojection / circularity 双重验收。
- **文件**：
  - `data/h36m_true_gt/s_01/05/06/07/08_acts_*_multiview_m.npz`（训练）
  - `data/h36m_true_gt/s_09/11_acts_*_multiview_m.npz`（测试）
  - Manifest：`configs/splits/h36m_true_gt_standard.yaml`
- **已知数字**：S9/S11 上 DLT baseline 约 25–34 mm，PA-MPJPE 约 23–32 mm；Iskakov 达到 combined 23.35 mm。
- **下一步**：v25 medium 已跑完，需先诊断其在真 GT 上发散的根因，再决定是否继续迭代 v25/v80/v57。

### P0-2 MPI-INF-3DHP 真实检测 2D 缺失

- **需要**：原始视频帧 `imageSequence/` + 真实 2D 检测器（CPN/HRNet/OpenPose/MediaPipe）。
- **当前状态**：A800 与本地均无 `imageSequence/`；现有 fallback 是 GT 2D + 2 px 高斯噪声 + 固定 confidence 0.81，**不能用于标准协议**。
- **下一步**：下载 MPI 原始数据并解压到 `data/webbridge/mpi_inf_3dhp/raw/S*/Seq*/imageSequence/`，运行 `scripts/generate_mpi_detected_2d.py` 的 `--detector <real>` 分支。

### P0-3 标准 SOTA 基线未完全复现

- Iskakov ICCV 2019 已在 H36M 和 Shelf/Campus 真 GT 上复现。
- VoxelPose、MVPose 等其他 SOTA 尚未跑通。
- 下一步：统一协议下补充更多 SOTA 基线。

---

## 4. 正在运行的后台任务

| 任务 | Agent / 来源 | 状态 | 说明 |
|---|---|---|---|
| H36M true-GT v25 medium | agent-51 | ✅ 已结束 | best epoch 2 = 72.80 mm，epoch 8 发散至 207.62 mm |
| H36M true-GT v80 medium | — | ✅ 已结束 | best epoch 4 = 39.98 mm，epoch 8 发散至 133.71 mm |
| H36M true-GT v57 medium | — |  状态未确认 | `nvidia-smi` 未观察到 GPU 训练进程；无输出 log |
| AIST++ 集成 v25/v80 | agent-67 | 已结束/空闲 | 仅完成 smoke 验证；未占用 GPU |

- **当前 GPU 空闲**：`nvidia-smi` 显示 RTX 4090 利用率 ~22–35%、内存 ~2.0 GB，无 `python.exe` GPU 训练进程。启动新任务前请先确认 `nvidia-smi` 仍保持空闲。
- 不要重复启动已结束的 v25 medium。

---

## 5. 建议的接力工作清单（按优先级）

1. **综合诊断 v25/v80/v57 在 H36M 真 GT 上的过拟合模式（P0）**
   - v25 medium 已结束（best 72.80 mm，epoch 2；发散至 207.62 mm）。
   - v80 medium 已结束（best 39.98 mm，epoch 4；发散至 133.71 mm）。
   - v57 medium 状态未确认；`nvidia-smi` 未观察到 GPU 训练进程，需先确认其实际状态再补充到 `docs/results_true_gt_h36m.md`。
   - 共同模式：所有 learned model 在 epoch 1–4 达到最佳，随后 train loss 继续下降但 val MPJPE 上升，属 classic overfitting。
   - 根因已初步定位（见 `docs/v25_divergence_diagnosis.md`）：`train_samples` 过少（1024/epoch，仅 64 steps）、缺少 weight decay/early stopping、 augmentation 过强、lr 偏高。
   - 产出：在 `docs/v25_true_gt_failure_mode.md`（或新建 `docs/true_gt_overfitting_diagnosis.md`）记录 v25/v80/v57 的曲线对比与统一修复方案。

2. **跑 v25/v80 短 epoch / 消融 smoke，验证修复方向**
   - 基于 `docs/v25_divergence_diagnosis.md` 建议的修复（`train_samples 4096`、`weight_decay 1e-4`、`lr 5e-4`、early stopping、降低 outlier 概率）跑 2–3 epoch smoke。
   - 目标：验证 val MPJPE 在 epoch 2 后不再单调上升。
   - 待 v57 结束后，用同一修复方案对 v57 做 smoke 验证，比较 v25/v80/v57 的起点差异。

3. **在 v57 结果落地后决定下一轮 medium 优先级（P1）**
   - 若 v57 优于 v80（< 39.98 mm）：优先对 v57 应用修复方案跑 medium，确认 DC-PSC 模块在真 GT 上的收益。
   - 若 v57 不优于 v80：优先修复 v80 的过拟合，因为 v80 当前已是最强 learned baseline。
   - 若修复后 v80/v57 仍追不上 Iskakov/DLT，则转向混合数据集 / 跨域训练或降低模型容量。

4. **MPI 真实检测 2D（P1）**
   - 下载 MPI `imageSequence/`。
   - 运行真实检测器并生成 `mpi_inf_3dhp_detected_2d/` 真实检测 .npz。
   - 复测 DLT baseline 与 v25/v80 在 MPI 标准协议上的表现。

5. **AIST++ 集成（P2）**
   - 检查 `agent-67` 结果（smoke 已完成）。
   - 将 AIST++ 加入跨域训练 mix，创建 manifest。

6. **复现/补充更多 SOTA（P2）**
   - VoxelPose、MVPose 等。

7. **重写论文方向（P2）**
   - 修正编造引用（`Iskakov` not `Iskandar`，`Ray-attention` 不存在等）。
   - 将卖点从“绝对精度”改为“稀疏视角 / 跨域鲁棒性”。
   - 更新 `docs/` 中相关故事文档。

---

## 6. 执行约束

- **A800 / Docker 只读**：可 `ssh a800-D` 查看文件，禁止启动/重启 Docker 或 tmux 训练。
- **GPU 使用**：仅本地 RTX 4090 用于 smoke/diagnostic；**不要同时跑多个训练进程**（当前 GPU 空闲；启动新任务前请先确认 `nvidia-smi` 无 `python.exe` GPU 训练进程）。
- **Swarm**：一次最多启用 15 个子代理；优先把独立任务并行化。
- **数据**：在没有真 GT / 真实检测 2D 之前，不要把 smoke 数字用于模型选择。

---

## 7. 完成标准（供设为 /goal）

- [ ] H36M 使用真 GT 3D 重新生成标准协议 .npz，且 DLT 与 v25 在该协议上得到合理 MPJPE（15–30 mm）。
- [ ] MPI-INF-3DHP 使用真实检测 2D 生成 .npz，并在该协议上重跑 DLT / v25 / v80 / v57 排行榜。
- [ ] 完成 Iskakov ICCV 2019 baseline 复现，指标与项目内其他方法对齐。
- [ ] 重写论文卖点与引用，提交到 MPI 官方测试服务器（如适用）。

---

## 8. 快速入口命令

```bash
# 诊断任意 .npz 是否循环
python scripts/diagnose_circular_labels.py data/webbridge/shelf_campus_detected/shelf_seq1_val_detected_m.npz

# true-GT 重投影一致性审计
python scripts/check_true_gt_reprojection.py data/h36m_true_gt/s_09_act_02_multiview_m.npz

# H36M true-GT Iskakov 基线
python experiments/train_iskakov_baseline_shelf_campus.py \
    --protocol h36m --epochs 10 --batch_size 8 --lr 1e-3 --hidden_dim 32 \
    --train_samples_per_epoch 4096 \
    --log_path outputs/iskakov_h36m_true_gt.log \
    --ckpt_path outputs/iskakov_h36m_true_gt.pth

# v25 true-GT smoke
bash scripts/run_v25_h36m_true_gt_medium_local_4090.sh

# 查看后台任务状态
TaskList(active_only=False)
```
