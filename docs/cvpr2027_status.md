# MotionFlow-MultiView CVPR 2027 Status (2026-08-12 ~15:15 UTC)

## 1. 核心结论

- **v85 训练已完成**，但 no-fallback 稀疏视角评估显示 **k<4 仍然灾难性**：k=2 约 2310 mm，k=3 约 1119 mm。随机 view dropout 单独还不足以解决稀疏视角问题。
- **v86 状态不明**：当前进程列表中不可见，需要检查日志/输出来确认是完成、崩溃还是被取代。
- **GPU 政策违规**：GPU 6/7 上出现了其他项目进程（LuxTTS、Mega-ASR、`.venv-cu130-a800` 等）。**不要 kill 这些进程**，但需记录并上报违规。
- **数据地基已闭环**。H36M true-GT v2 审计完成，7 个 `.npz` 全部非循环（direct MJE 13.66–33.83 mm），可直接用于模型选择。
- **真 GT v2 leaderboard 已建立**：DLT conf-weighted **25.67 mm**、unweighted **28.77 mm**、RANSAC **26.47 mm**；v25/v86 smoke 在 v2 上分别为 **67.71 mm** 和 **78.41 mm**。
- **稀疏视角仍是关键卖点**：v85 no-fallback 失败后，下一步是先跑 v85 DLT-fallback 评估，再决定是否需要 count-conditioning 增强或独立稀疏 head。
- **跨域与提交准备并行**：MPI 官方服务器提交包已备妥；下一步是同步 v2 到 A800 并重跑 learned leaderboard。

## 2. 数据地基现状

| 数据集 | 真 GT / 非循环？ | 状态 |
|---|---|---|
| H36M true GT v2 | ✅ | `data/h36m_true_gt_v2/` 审计完成；7 文件均非循环（direct MJE 13.66–33.83 mm）。v2 DLT: conf **25.67 mm** / unw **28.77 mm**；RANSAC **26.47 mm**。待同步到 A800。 |
| H36M true GT v1 | ️ | `data/h36m_true_gt/` 已对齐但存在偏差；逐步迁移到 v2。 |
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
| v82 + DLT fallback | 58.18 / 49.35 | 33.32 / 25.28 | 47.81 / 42.36 | k=2/3 DLT fallback，k=4  learned v82 |
| v85 no-fallback | 2310.27 / 2308.80 | 1119.45 / 1118.18 | 83.52 / 77.07 | **DONE**；k<4 仍灾难性；k=4 弱于 v82 |
| v86 no-fallback | — | — | — | **UNCERTAIN**；进程不可见，需检查 log/checkpoint |

- **关键结论**：即使原生随机 view dropout 训练，v85 在 k<4 时仍然崩溃。dropout 本身不够，需要更强的 count-conditioning 或独立稀疏 head。
- **下一步**：先跑 v85 DLT-fallback 评估，看 k=2/3 是否能在 fallback 下回到 v25/v82 水平；同时确认 v86 实际状态。
- 稀疏视角对比脚本已就绪：`scripts/compare_sparse_view_v85_v86.py`（或等效脚本），v85 DLT-fallback/v86 结果出来后可直接对比。

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
| **GPU 7** | v85 random view dropout H36M true-GT medium | **DONE** | `outputs/ablations/v85_random_view_dropout_medium_a800.log`；post-training eval suite 排队中 |
| **GPU 6** | v86 no-count-embedding ablation medium | **UNCERTAIN** | `outputs/ablations/v86_no_count_embedding_medium_a800.log`；当前进程不可见，需确认 |
| **GPU 6/7** | 其他项目进程（LuxTTS、Mega-ASR、`.venv-cu130-a800`） | **POLICY VIOLATION** | 不要 kill，但需记录/上报；MotionFlow 只应使用 GPU 6/7 |

- **不要停止或干扰上述其他项目进程**；由项目负责人协调解决 GPU 占用问题。
- v85 post-training eval suite（test-set eval + no-fallback + DLT-fallback）仍排队，待 GPU 空闲后自动在首个空闲 GPU 启动。
- v85 no-fallback 结果已出，下一步重点是触发并等待 v85 DLT-fallback 结果。

## 7. 关键阻塞

| 阻塞 | 影响 | 下一步 |
|---|---|---|
| **GPU 6/7 政策违规** | 其他项目进程占用 MotionFlow GPU，可能阻塞 v85 post-training eval | 记录并上报；不要 kill 其他进程；等待协调结果 |
| **稀疏视角 k<4 仍未解决** | v85 no-fallback k=2/3 灾难性 | 跑 v85 DLT-fallback；若仍差，设计 count-conditioning 增强或独立稀疏 head |
| **A800 磁盘 99% 满** | 仅剩 ~58 GB，无法启动大型新实验 | v85/v86 结果确认后运行 `scripts/cleanup_a800_safe.sh` dry-run |
| **H36M true-GT v2 待同步** | v2 `.npz` 仅在本地，未同步到 A800 | GPU/磁盘空间就绪后同步并重跑 learned leaderboard |

## 8. 下一阶段里程碑

1. **确认 v86 状态并跑 v85 DLT-fallback**
   - 检查 `outputs/ablations/v86_no_count_embedding_medium_a800.log` 和 checkpoint 是否存在，判断 v86 是完成、崩溃还是被取代。
   - 确保 v85 post-training eval suite（PID `2218949`）正常排队，GPU 空闲后启动 DLT-fallback 评估。
   - 对比 v85 DLT-fallback k=2/3/4 与 v25/v82 DLT-fallback 基线。

2. **处理 GPU 政策违规**
   - 记录当前 GPU 6/7 上的其他项目进程（LuxTTS、Mega-ASR、`.venv-cu130-a800`）。
   - 由项目负责人协调解决，不要自行 kill 进程。

3. **A800 清理**
   - 运行 `scripts/cleanup_a800_safe.sh` dry-run。
   - 优先删除 v83/v84 失败产物、重复 manifest、旧 variable-view 临时文件。

4. **同步 H36M true-GT v2 并重跑 learned leaderboard**
   - 将 `data/h36m_true_gt_v2/` 同步到 A800。
   - 在 v2 上重跑 DLT/Iskakov/v25/v46/v52/v57/v80/v81/v82/v85/v86，更新 `docs/results_true_gt_h36m.md`。

5. **MPI 官方服务器提交**
   - 提交包已准备；v2 leaderboard 稳定后上传并记录结果。

6. **SOTA 比较**
   - VoxelPose / MVPose / DLT configs 已就绪，待 GPU 6/7 空闲/协调后调度。

7. **论文重写**
   - 更新 `docs/paper_draft_icra_cvpr_2027.md` 的表格与引用。
   - 故事围绕：真 GT 诚实评估、稀疏视角鲁棒性、跨域泛化（AIST++/MPI）。

## 9. GPU / 训练资源状态

- **仅 GPU 6 和 GPU 7 可用**；GPU 0–5 为其他项目保留，严禁使用。
- 当前 GPU 6/7 存在其他项目进程，属于 GPU 政策违规，需协调处理。
- 启动任何新训练/评估前，必须确认 GPU 空闲且 `CUDA_VISIBLE_DEVICES` 为 6 或 7。
- `/mnt/nvme0n1p1/zhangzy/projects` 和 A800 Docker `motionflow` 服务为 **只读**。

## 10. 立即行动（下一步）

1. **确认 v86 实际状态**：检查 `outputs/ablations/v86_no_count_embedding_medium_a800.log` 和 checkpoint。
2. **跑 v85 DLT-fallback 评估**：确认 post-training eval suite 正常排队，GPU 空闲后启动。
3. **处理 GPU 政策违规**：记录并上报 GPU 6/7 上的其他项目进程，不要 kill。
4. **A800 磁盘清理**：运行 `scripts/cleanup_a800_safe.sh` dry-run，释放空间。
5. **同步 v2 并重跑 leaderboard**：磁盘/GPU 就绪后将 `data/h36m_true_gt_v2/` 同步到 A800。
6. **提交 MPI 官方服务器包并记录结果**。
7. **更新 `docs/paper_draft_icra_cvpr_2027.md` 的结果与卖点描述**。

---

> 当前最紧迫：**确认 v86 状态**、**跑 v85 DLT-fallback 评估**、**处理 GPU 6/7 政策违规**、**A800 磁盘空间**、**v2 同步与 leaderboard 重跑**。
