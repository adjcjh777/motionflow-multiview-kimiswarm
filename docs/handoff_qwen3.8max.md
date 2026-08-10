# MotionFlow-MultiView 接力目标（qwen3.8max）

> **目标**：修复数据地基，建立非循环评估协议，重建可发表水准的排行榜，锚定 CVPR 2027。  
> **当前日期**：2026-08-10  
> **仓库**：`D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`  
> **远程只读资源**：A800-D `/mnt/nvme0n1/zhangzy/projects` 与 `motionflow` Docker 容器仅供查看，禁止写入或启动训练。

---

## 1. 核心结论（fable5 建议）

本项目当前形式**不会收敛到可发表成果**。根因是数据地基坏了：

- **H36M**：现有 `data/h36m_hf/*.npz`、`data/webbridge/h36m_meters/*.npz`、A800 上的 `h36m_meters` 均为**循环标签**——`joints_3d` 是 `points_2d + cameras` 的 DLT 三角化结果，direct MJE ≈ 0 mm。模型学到的是“复现 DLT”，不是人体姿态。
- **Shelf/Campus**：旧版 `data/webbridge/shelf_campus/*.npz` 也是 GT 投影生成的循环标签。
- **MPI-INF-3DHP**：标签是真 3D，但目前训练用的是 GT 投影 2D，与文献标准的“检测 2D”协议不可比。

因此，**v25 及之前所有数字均不能用于模型选择**。卖点必须从“绝对 MPJPE 碾压”转向**稀疏视角 / 跨域鲁棒性**。

---

## 2. 本阶段已完成

| 任务 | 状态 | 关键产出 |
|------|------|----------|
| 停止 A800 训练队列 | ✅ | 已确认 A800-D 上无训练进程 |
| 闭环验证 H36M 循环标签 | ✅ | `scripts/diagnose_circular_labels.py` direct MJE ≈ 0 mm |
| 修复混合 loader 的 domain embedding 越界 | ✅ | `experiments/train_omniview_fusion_v5_webbridge_multi.py` 自动根据 manifest 最大 dataset id 设置 `num_domains` |
| 生成非循环 Shelf/Campus .npz | ✅ | `experiments/build_shelf_campus_canonical_from_detection.py` |
| 更新非循环 smoke manifest | ✅ | `configs/splits/shelf_campus_noncircular_smoke.yaml` |
| 运行 v25 true-GT Shelf/Campus smoke | ✅ | best val MPJPE = **371.66 mm**；同份数据 DLT baseline ≈ **130 mm** |
| A800 只读数据侦察 | ✅ | 未找到 H36M 真 GT (`PosesD3_Positions`) 与 MPI `imageSequence`；A800 上 H36M .npz 仍为循环标签 |
| MPI 检测 2D fallback 审计 | ✅ | `data/webbridge/mpi_inf_3dhp_detected_2d/` 是 GT 2D + 2 px 高斯噪声 + 固定 confidence 0.81，**不能替代真实检测 2D** |
| Git 清理 | ✅ | 删除旧 tmp/outputs 约 240 MB |

### 2.1 关键文件路径

- 非循环 Shelf/Campus .npz：
  - `data/webbridge/shelf_campus_detected/shelf_seq1_train_detected_m.npz`
  - `data/webbridge/shelf_campus_detected/shelf_seq1_val_detected_m.npz`
  - `data/webbridge/shelf_campus_detected/campus_seq1_train_detected_m.npz`
  - `data/webbridge/shelf_campus_detected/campus_seq1_val_detected_m.npz`
- 诊断脚本：`scripts/diagnose_circular_labels.py`
- 转换脚本：`experiments/build_shelf_campus_canonical_from_detection.py`
- 修复 domain embedding 的训练入口：`experiments/train_omniview_fusion_v5_webbridge_multi.py`
- H36M 转换入口（待接入真 GT）：`motionflow_mv/data/webbridge_loader.py:182` / `experiments/prepare_h36m_multiview.py`

### 2.2 已完成的后台 benchmark（true-GT Shelf/Campus detected）

| 方法 | Val MPJPE / error (mm) | 备注 |
|------|------------------------|------|
| DLT baseline – direct MJE | **134.43** | 用 .npz 里的 2D + cameras 重新三角化 vs 真 3D |
| DLT baseline – root-aligned MPJPE | **122.37** | Procrustes 对齐后的 DLT 误差 |
| v25 | 430.67 | 3-epoch smoke |
| v80 | **408.58** | 3-epoch smoke；学习模型中最好 |
| v57 | 424.63 | 3-epoch smoke |

- 所有运行无 NaN/inf/crash。
- 在该 true-GT、稀疏视角（Shelf 5 views / Campus 3 views）数据上，**朴素 DLT 显著优于 3-epoch 学习模型**。
- 学习模型中 **v80（视图可靠性加权）> v57 > v25**，说明 v80 的方向对稀疏跨域场景最有价值，但仍需更多训练/调优。

---

## 3. 剩余关键阻塞（必须解除才能继续模型迭代）

### P0-1 H36M 真 3D GT 缺失

- **需要**：Human3.6M 官方 mocap 世界坐标 `PosesD3_Positions`（或同等级真 3D）。
- **当前状态**：本地、A800、Docker 均未找到。
- **下一步**：下载/拷贝到 `data/h36m_true_gt/`；修改 `motionflow_mv/data/webbridge_loader.py` 与 `experiments/prepare_h36m_multiview.py`，用真 GT 替换 `_triangulate_joints`；重新生成标准协议 `S1,5,6,7,8 → S9/S11` 的 .npz。

### P0-2 MPI-INF-3DHP 真实检测 2D 缺失

- **需要**：原始视频帧 `imageSequence/` + 真实 2D 检测器（CPN/HRNet/OpenPose/MediaPipe）。
- **当前状态**：A800 与本地均无 `imageSequence/`；现有 fallback 是 GT 2D + 噪声，不能用于标准协议。
- **下一步**：下载 MPI 原始数据并解压到 `data/webbridge/mpi_inf_3dhp/raw/S*/Seq*/imageSequence/`，运行 `scripts/generate_mpi_detected_2d.py` 的 `--detector <real>` 分支。

### P0-3 标准 SOTA 基线未复现

- Iskakov et al. ICCV 2019 baseline、VoxelPose 等均未真实跑通。
- 下一步：在修复数据后，用统一协议重跑 DLT / RANSAC / v25 / v46 / v57 / v80，并加入 Iskakov 复现。

---

## 4. 建议的接力工作清单（按优先级）

1. **接管后台 benchmark**  
   - 检查 `agent-47` 结果。  
   - 若 v80/v57 跑完，整理 true-GT Shelf/Campus 排行榜；若未跑完，继续跑完并汇总。

2. **获取 H36M 真 GT**  
   - 优先从官方 Human3.6M release 取得 `PosesD3_Positions`（或项目已有渠道）。  
   - 写入 `data/h36m_true_gt/`。

3. **接入 H36M 真 GT 并重建协议**  
   - 修改 `motionflow_mv/data/webbridge_loader.py` 的 Shelf/Campus 风格转换逻辑：从真 GT 读取 3D，而非 `_triangulate_joints`。  
   - 重新生成 H36M 标准训练/测试 .npz（S1,5,6,7,8 训练；S9/S11 测试）。  
   - 运行 DLT/v25 验证数字落在 15–30 mm 区间。

4. **获取/生成 MPI 真实检测 2D**  
   - 下载 MPI `imageSequence`。  
   - 运行真实检测器并生成 `mpi_inf_3dhp_detected_2d/` 真实检测 .npz。  
   - 复测 DLT baseline 与 v25/v80 在 MPI 标准协议上的表现。

5. **复现 Iskakov ICCV 2019 baseline**  
   - 在统一协议（H36M S1,5,6,7,8→S9/S11；MPI detected-2D；Shelf/Campus true-GT）下复现并记录指标。

6. **重写论文方向**  
   - 修正编造引用（`Iskakov` not `Iskandar`，`Ray-attention` 不存在等）。  
   - 将卖点从“绝对精度”改为“稀疏视角 / 跨域鲁棒性”。  
   - 更新 `docs/` 中相关故事文档。

---

## 5. 执行约束

- **A800 / Docker 只读**：可 `ssh a800-D` 查看文件，禁止启动/重启 Docker 或 tmux 训练。
- **GPU 使用**：仅本地 RTX 4090 用于 smoke/diagnostic；不要同时在 4090 上跑多个训练进程。
- **Swarm**：一次最多启用 15 个子代理；优先把独立任务并行化。
- **数据**：在没有真 GT / 真实检测 2D 之前，不要把 smoke 数字用于模型选择。

---

## 6. 完成标准（供设为 /goal）

- [ ] H36M 使用真 GT 3D 重新生成标准协议 .npz，且 DLT 与 v25 在该协议上得到合理 MPJPE（15–30 mm）。
- [ ] MPI-INF-3DHP 使用真实检测 2D 生成 .npz，并在该协议上重跑 DLT / v25 / v80 / v57 排行榜。
- [ ] 完成 Iskakov ICCV 2019 baseline 复现，指标与项目内其他方法对齐。
- [ ] 重写论文卖点与引用，提交到 MPI 官方测试服务器（如适用）。

---

## 7. 快速入口命令

```bash
# 诊断任意 .npz 是否循环
python scripts/diagnose_circular_labels.py data/webbridge/shelf_campus_detected/shelf_seq1_val_detected_m.npz

# 重建 Shelf/Campus 非循环 .npz
python experiments/build_shelf_campus_canonical_from_detection.py

# v25 true-GT smoke
bash scripts/run_v25_shelf_campus_noncircular_smoke_local_4090.sh

# 查看后台 benchmark 状态
TaskList(active_only=False)
TaskOutput("agent-a2wr3anz")
```
