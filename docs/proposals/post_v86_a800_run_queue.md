# v86 完成后 A800 任务执行清单

> **状态快照（2026-08-13 ~02:41 UTC）**
> - v86 no-count-embedding ablation 正在 A800 GPU 6 上运行，预计 early stop 在 epoch 6–8。
> - v85 random-view-dropout 训练已完成，checkpoint 已保存。
> - GPU 7 仍被外部项目占用，因此所有新任务只能使用 GPU 6/7，且必须等 GPU 6 或 7 空闲。
> - A800 `/mnt/nvme0n1p1` 使用率约 98%，启动任何大写入前请先做 `scripts/cleanup_a800_safe.sh` dry-run。
>
> **注意：此文档仅用于计划与监控，不要在当前 session 中启动任何 A800 任务。**

---

## 1. 已有 v2 结果排查

| 模型 | v1 true-GT 结果 | v2 true-GT 训练脚本 | v2 true-GT 输出 | 结论 |
|------|------------------|--------------------|------------------|------|
| v81 temporal-pose-attention | 有：`v81_true_gt_h36m_medium_a800`（val 38.62 mm） | 无 | 无 | **需在 v2 上重跑** |
| v82 multi-scale temporal-pose-attention | 有：`v82_true_gt_h36m_medium_a800`（val 39.58 mm） | 无 | 无 | **需在 v2 上重跑** |
| v46 sparse-view geometry (SVG) | 有 v1 test 结果 | 有：`scripts/run_v46_true_gt_v2_medium_a800.sh` | 无 | **需跑 v2 medium** |
| v52 UWT | 有 v1 test 结果 | 有：`scripts/run_v52_true_gt_v2_medium_a800.sh` | 无 | **需跑 v2 medium** |
| v57 DC-PSC | 有 v1 结果 | 有：`scripts/run_v57_true_gt_v2_medium_a800.sh` | 无 | **需跑 v2 medium** |

**结论：** v81/v82/v46/v52/v57 都还没有在 `data/h36m_true_gt_v2/` 上训练或评估过，都需要在 v2 协议下重跑。

---

## 2. 任务列表（按优先级排序）

### P0：v86 完成后立即自动/手动触发

| 优先级 | 任务 | 脚本路径 | 预计时间 | GPU / 显存 | 依赖 | 目的 |
|--------|------|----------|----------|------------|------|------|
| P0 | v25 true-GT v2 test-set 评估 | `scripts/run_v25_true_gt_v2_test_a800.sh` | 1–2 h | GPU 6/7，~8 GB | 依赖 `outputs/ablations/v25_true_gt_v2_medium_a800.pth`（已存在） | 获取 v25 在 H36M S9/S11 上的官方 test MPJPE/PA-MPJPE，补充 True-GT v2 Leaderboard |
| P0 | v85 DLT-fallback 可变视角评估 | `scripts/eval_variable_views_v85_random_view_dropout_medium_a800_dlt_fallback.sh`（由 `scripts/launch_v85_dlt_fallback_after_v86.sh` 自动触发） | 1–2 h | GPU 6/7，~8 GB | 依赖 `outputs/ablations/v85_random_view_dropout_medium_a800.pth`（已存在） | 判断 v85 random dropout 在 k=2/3 时是否需要 DLT fallback；补充稀疏视角实验 |

### P1：v2 medium 训练补齐

| 优先级 | 任务 | 脚本路径 | 预计时间 | GPU / 显存 | 依赖 | 目的 |
|--------|------|----------|----------|------------|------|------|
| P1 | v81 true-GT v2 medium | 需新建 `scripts/run_v81_true_gt_v2_medium_a800.sh` | 8–12 h | GPU 6/7，~15–25 GB | 依赖 v2 数据 `configs/splits/h36m_true_gt_v2_standard.yaml`；参考 v1 脚本 `scripts/run_v81_true_gt_h36m_medium_a800.sh` | 验证 temporal-pose-attention 在 v2 真实 3D 协议下的性能，与 v1 结果对比 |
| P1 | v82 true-GT v2 medium | 需新建 `scripts/run_v82_true_gt_v2_medium_a800.sh` | 8–12 h | GPU 6/7，~15–25 GB | 依赖 v2 数据；参考 v1 脚本 | 验证 multi-scale temporal-pose-attention 在 v2 协议下的性能 |
| P1 | v46 true-GT v2 medium | `scripts/run_v46_true_gt_v2_medium_a800.sh` | 8–12 h | GPU 6/7，~15–25 GB | 依赖 v2 数据 | 验证 sparse-view geometry (SVG) 在 v2 协议下的表现 |
| P1 | v52 true-GT v2 medium | `scripts/run_v52_true_gt_v2_medium_a800.sh` | 8–12 h | GPU 6/25 GB | 依赖 v2 数据 | 验证 uncertainty-weighted triangulation (UWT) 在 v2 协议下的表现 |
| P1 | v57 true-GT v2 medium | `scripts/run_v57_true_gt_v2_medium_a800.sh` | 8–12 h | GPU 6/7，~15–25 GB | 依赖 v2 数据 | 验证 domain-conditional physical-space calibration (DC-PSC) 在 v2 协议下的表现 |

### P2：SOTA 基线与 MPI 官方提交

| 优先级 | 任务 | 脚本路径 | 预计时间 | GPU / 显存 | 依赖 | 目的 |
|--------|------|----------|----------|------------|------|------|
| P2 | Iskakov ICCV 2019 v2 baseline | `scripts/run_iskakov_true_gt_v2_baseline_a800.sh` | 4–8 h | GPU 6/7，~10 GB | 依赖 v2 数据 | 重跑 SOTA learnable triangulation 在 v2 协议下的结果，作为论文主要 baseline（v1 报告 23.40 mm） |
| P2 | VoxelPose v2 baseline | `scripts/run_voxelpose_true_gt_v2_a800.sh` | 12–24 h（含数据转换 + 训练） | GPU 6/7，~20–30 GB | 依赖 v2 数据、conda env `voxelpose_py38_pt112`、上游 `models/voxelpose-pytorch` | 重跑 SOTA voxel-based 多视角 baseline 在 v2 协议下的结果 |
| P2 | MPI-INF-3DHP 官方服务器提交 | `scripts/run_mpi_submission_a800.sh` | 2–4 h（推理 + 打包） | GPU 6/7，~10 GB | 依赖选定 checkpoint 和 MPI test set 转换结果；详见 `docs/mpi_server_submission_plan.md` | 生成 MPI-INF-3DHP 官方 leaderboard 提交文件 |

### P3：跨数据集评估

| 优先级 | 任务 | 脚本路径 | 预计时间 | GPU / 显存 | 依赖 | 目的 |
|--------|------|----------|----------|------------|------|------|
| P3 | AIST++ 跨数据集评估 | `scripts/run_v25_aistpp_eval_a800.sh` | 1–2 h | GPU 6/7，~8 GB | 依赖 `v25_true_gt_v2_medium_a800.pth` | 测试 v25 v2 在 AIST++ 上的跨域泛化能力 |
| P3 | Shelf/Campus 跨数据集评估 | `scripts/run_v25_shelf_campus_eval_a800.sh` | 1–2 h | GPU 6/7，~8 GB | 依赖 `v25_true_gt_v2_medium_a800.pth` | 测试 v25 v2 在 Shelf/Campus 检测数据上的泛化能力 |

---

## 3. 建议启动顺序

### 阶段 A：v86 刚完成时（GPU 6 空闲，GPU 7 仍被占用）

1. **确认 v86 完成**（tmux session `v86_no_count_embedding` 结束，`v86_no_count_embedding_medium_a800.pth` 已保存）。
2. **自动触发**：`scripts/launch_v85_dlt_fallback_after_v86.sh` 应该已经排队，会自动在 GPU 6 上启动 **v85 DLT-fallback 可变视角评估**。
3. **手动启动**：`scripts/run_v25_true_gt_v2_test_a800.sh` 在 GPU 6 上运行（如果 GPU 6 被 v85 eval 占用，则等它结束）。
4. **（可选）** 若 GPU 7 已空，可将 v25 test eval 放到 GPU 7。

### 阶段 B：P1 v2 medium 训练（GPU 7 恢复后或 v85/v25 eval 完成后）

按以下顺序排队执行，每次只占用一个 GPU：

1. v81 true-GT v2 medium（需先创建脚本）
2. v82 true-GT v2 medium（需先创建脚本）
3. v46 true-GT v2 medium
4. v52 true-GT v2 medium
5. v57 true-GT v2 medium

> **建议**：如果 GPU 时间紧张，至少先跑 v81 和 v82，因为 v81 是 v82 的基础；v46/v52/v57 选一个即可作为后续消融的锚点。

### 阶段 C：P2 SOTA 基线与 MPI 提交

1. Iskakov v2 baseline
2. VoxelPose v2 baseline
3. MPI-INF-3DHP 官方提交（使用 v25 或 v85 的 checkpoint，根据当时最好的模型决定）

### 阶段 D：P3 跨数据集评估

1. AIST++ 跨数据集评估
2. Shelf/Campus 跨数据集评估

---

## 4. 任务目的简述

- **v25 true-GT v2 test-set 评估**：v25 在 H36M true-GT v2 协议下的完整 test 指标是论文核心数字，目前只有 val 31.41 mm，需要 S9/S11 test MPJPE 与 PA-MPJPE。
- **v85 DLT-fallback 可变视角评估**：判断 random view dropout 训练是否缓解了 k<4 灾难性失败；与 v81/v82/v86 的 DLT-fallback 结果对比，决定是否需要更强的稀疏视角头。
- **v81/v82 v2 medium**：验证 temporal / multi-scale temporal pose attention 在真实 3D 协议下是否仍优于 v25 baseline。
- **v46/v52/v57 v2 medium**：验证稀疏视角、不确定性感知三角测量和域条件物理空间校准在 v2 协议下的独立贡献。
- **Iskakov / VoxelPose v2 baseline**：获取 corrected true-GT v2 协议下的 SOTA 对比数字，替换原 circular-label 结果。
- **MPI 官方服务器提交**：将最佳模型提交至 MPI-INF-3DHP 官方 leaderboard，获得跨数据集泛化的第三方指标。
- **AIST++ / Shelf/Campus 跨数据集评估**：验证模型在 H36M 以外数据集上的泛化能力，支撑 paper 的 cross-domain robustness 叙事。

---

## 5. 当前阻塞与注意事项

1. **GPU 7 仍被外部项目占用**。在 GPU 7 释放前，所有任务只能串行使用 GPU 6。
2. **v81/v82 v2 脚本尚未创建**。需要在跑之前从 `scripts/run_v81_true_gt_h36m_medium_a800.sh` 和对应的 v82 脚本修改出 v2 版本，核心改动：
   - `--mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml`
   - 输出路径改为 `v81_true_gt_v2_medium_a800` / `v82_true_gt_v2_medium_a800`
   - 默认 `CUDA_VISIBLE_DEVICES=6`（GPU 政策）
3. **磁盘空间紧张**。`/mnt/nvme0n1p1` 约 98% 满。任何新的 medium 训练前都应先跑 `bash scripts/cleanup_a800_safe.sh --dry-run`。
4. **A800 只读/监控**。当前 session 只创建本计划文档，不启动任何 A800 训练或评估。
