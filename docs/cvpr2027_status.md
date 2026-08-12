# MotionFlow-MultiView CVPR 2027 Status (2026-08-12 ~15:07 UTC)

## 1. 核心结论

- **数据地基已闭环**。H36M true-GT v2 审计完成，7 个 `.npz` 全部非循环（direct MJE 13.66–33.83 mm），可直接用于模型选择。
- **真 GT v2 leaderboard 已建立**：DLT conf-weighted **25.67 mm**、unweighted **28.77 mm**、RANSAC **26.47 mm**；v25/v86 smoke 在 v2 上分别为 **67.71 mm** 和 **78.41 mm**。
- **稀疏视角仍是关键卖点**：v85 随机 view dropout 和 v86 no-count-embedding 正在 A800 GPU 7/6 训练；结果将决定是否需要在 count-conditioning 或独立稀疏 head 上继续投入。
- **跨域与提交准备并行**：MPI 官方服务器提交包已备妥；下一步是同步 v2 到 A800 并重跑 learned leaderboard。

## 2. 数据地基现状

| 数据集 | 真 GT / 非循环？ | 状态 |
|---|---|---|
| H36M true GT v2 | ✅ | `data/h36m_true_gt_v2/` 审计完成；7 文件均非循环（direct MJE 13.66–33.83 mm）。v2 DLT: conf **25.67 mm** / unw **28.77 mm**；RANSAC **26.47 mm**。待同步到 A800。 |
| H36M true GT v1 | ⚠️ | `data/h36m_true_gt/` 已对齐但存在偏差；逐步迁移到 v2。 |
| MPI-INF-3DHP | ✅ 真 GT | RTMPose 真实检测 2D 已完成（16/16 `.npz`）；DLT baseline **115.09 mm** / PA-MPJPE **132.68 mm**。官方服务器提交包已准备。 |
| Shelf/Campus detected | ✅ | `data/webbridge/shelf_campus_detected/` 已就绪。 |
| AIST++ | ✅ | 1,408 个 canonical `.npz` 已就绪；AIST++-only medium fast v2 完成；H36M cross-eval **93.94 mm**。 |
| 3DPW | ⚠️ | pseudo 循环；actual 单目，不可三角化，暂不用于排行榜。 |

## 3. H36M true-GT v2 排行榜（S1,5,6,7,8 → S9/S11）

| 方法 | Combined MPJPE (mm) | PA-MPJPE (mm) | 备注 |
|---|---:|---:|---|
| **Iskakov ICCV 2019** | **23.40** | — | 几何方法 leader |
| DLT (conf-weighted) | **25.67** | — | v2  frozen ref |
| DLT (unweighted) | **28.77** | — | v2  frozen ref |
| RANSAC/conf-DLT | **26.47** | — | v2  reproducible |
| v25 stability (3-epoch smoke) | **67.71** | — | v2  local smoke；待 A800 medium 重跑 |
| v86 no-count-embedding (2-epoch smoke) | **78.41** | — | v2  local smoke；待 A800 medium 重跑 |

- 详细表格与日志见 `docs/results_true_gt_h36m.md`。
- v2 的 v25/v81/v82/v85/v86/v46/v52/v57/v80 等待 A800 空闲后重跑。

## 4. 稀疏视角（variable view）结果

| 方法 | k=2 S9/S11 | k=3 S9/S11 | k=4 S9/S11 | 备注 |
|---|---|---|---|---|
| v25 + DLT fallback | 58.18 / 49.35 | 33.32 / 25.28 | 116.98 / 110.58 | k<4 用 conf-DLT，k=4 用 learned model |
| v82 + DLT fallback | 58.18 / 49.35 | 33.32 / 25.28 | 47.81 / 42.36 | k=2/3 DLT fallback，k=4 learned v82 |
| v85 no-fallback | — | — | — | **RUNNING** on GPU 7；首个原生随机 view dropout 训练模型 |
| v86 no-fallback | — | — | — | **RUNNING** on GPU 6；ablates active-view-count embedding |

- **稀疏视角对比脚本已就绪**：`scripts/compare_sparse_view_v85_v86.py`（或等效脚本），v85/v86 训练完成后可直接对比 k=2/3/4。
- 关键结论：当前非 dropout 模型在 k<4 时仍会崩溃；v85/v86 的结果将决定随机 view dropout 是否能从根本上解决该问题。

## 5. 跨域结果

| 训练数据 | 测试数据 | Combined MPJPE (mm) | 备注 |
|---|---:|---:|---|
| AIST++ only | AIST++ val | 91.43 | best @ epoch 4 |
| AIST++ only | H36M true-GT S9/S11 | **93.94** | S9 98.17 / S11 89.70 |
| AIST++ full | AIST++ | DLT 15.93 mm weighted / 38.11 mm unweighted | frozen ref |

- 详细结果见 `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json`、`outputs/aistpp_full_dlt_baseline_a800.json`。

## 6. 在飞工作

| GPU | 任务 | 状态 | 输出 / 备注 |
|---|---|---|---|
| **GPU 7** | v85 random view dropout H36M true-GT medium | **RUNNING** | `outputs/ablations/v85_random_view_dropout_medium_a800.log` |
| **GPU 6** | v86 no-count-embedding ablation medium | **RUNNING** | `outputs/ablations/v86_no_count_embedding_medium_a800.log`；A800 config/script 已就绪 |

- **不要停止或干扰上述进程。**
- v86 A800 配置：`configs/ablations/v86_no_count_embedding_medium_a800.yaml`；脚本：`scripts/v86_no_count_embedding_medium_a800.sh`。
- v85 post-training eval suite（test-set eval + no-fallback + DLT-fallback）仍排队，v85 训练完成后自动在首个空闲 GPU 启动。

## 7. 关键阻塞

| 阻塞 | 影响 | 下一步 |
|---|---|---|
| **v85/v86 训练中** | GPU 6/7 被占，无法启动 SOTA/medium 重跑 | 等待训练完成；不要抢占 GPU 6/7 |
| **稀疏视角 k<4 失败仍未解决** | 非 dropout 模型 k=2/3  catastrophic | 等待 v85/v86 评估结果 |
| **A800 磁盘 99% 满** | 仅剩 ~58 GB，无法启动大型新实验 | v85/v86 完成后运行 `scripts/cleanup_a800_safe.sh` dry-run |
| **H36M true-GT v2 待同步** | v2 `.npz` 仅在本地，未同步到 A800 | GPU 空闲后同步并重跑 learned leaderboard |

## 8. 下一阶段里程碑

1. **等待 v85/v86 完成**
   - GPU 7/6 训练完成后自动/手动触发 post-training eval。
   - 对比 v85/v86 的 k=2/3/4 与 v25/v82 DLT-fallback 基线。

2. **A800 清理**
   - 运行 `scripts/cleanup_a800_safe.sh` dry-run。
   - 优先删除 v83/v84 失败产物、重复 manifest、旧 variable-view 临时文件。

3. **同步 H36M true-GT v2 并重跑 learned leaderboard**
   - 将 `data/h36m_true_gt_v2/` 同步到 A800。
   - 在 v2 上重跑 DLT/Iskakov/v25/v46/v52/v57/v80/v81/v82/v85/v86，更新 `docs/results_true_gt_h36m.md`。

4. **MPI 官方服务器提交**
   - 提交包已准备；v2 leaderboard 稳定后上传并记录结果。

5. **SOTA 比较**
   - VoxelPose / MVPose / DLT configs 已就绪，待 GPU 6/7 空闲后调度。

6. **论文重写**
   - 更新 `docs/paper_draft_icra_cvpr_2027.md` 的表格与引用。
   - 故事围绕：真 GT 诚实评估、稀疏视角鲁棒性、跨域泛化（AIST++/MPI）。

## 9. GPU / 训练资源状态

- **仅 GPU 6 和 GPU 7 可用**；GPU 0–5 为其他项目保留，严禁使用。
- 启动任何新训练/评估前，必须确认 GPU 空闲且 `CUDA_VISIBLE_DEVICES` 为 6 或 7。
- `/mnt/nvme0n1p1/zhangzy/projects` 和 A800 Docker `motionflow` 服务为 **只读**。

## 10. 立即行动（下一步）

1. **等待 v85/v86 训练完成**，不要抢占 GPU 6/7。
2. v85/v86 完成后整理稀疏视角结果，判断 k<4 问题是否解决。
3. 运行磁盘清理 dry-run，释放 A800 空间。
4. 同步 H36M true-GT v2 到 A800，重跑 learned leaderboard。
5. 提交 MPI 官方服务器包并记录结果。
6. 更新 `docs/paper_draft_icra_cvpr_2027.md` 的结果与卖点描述。

---

> 当前最紧迫：**v85/v86 训练结果**、**A800 磁盘空间**、**v2 同步与 leaderboard 重跑**。
