# CVPR 2027 投稿里程碑清单

> **项目：** MotionFlow-MultiView
> **当前日期：** 2026-08-12
> **目标会议：** CVPR 2027
> **预计截稿：** 2026-11-07（abstract） / 2026-11-14（full paper）
> **剩余时间：** 约 **13 周**（91 天）

---

## 1. 剩余时间概览

| 里程碑 | 起止日期 | 时长 | 核心目标 |
|---:|:---|:---:|:---|
| M1 数据地基收尾 | 08/12 – 08/18 | 1 周 | v2 标签同步、验证、基准复现 |
| M2 基线重跑 | 08/19 – 09/01 | 2 周 | v25 v2、v86 A800、DLT/RANSAC/Iskakov |
| M3 SOTA 比较 | 09/02 – 09/15 | 2 周 | VoxelPose、MVPose、DLT 完整对比 |
| M4 消融与鲁棒性曲线 | 09/16 – 09/29 | 2 周 | 稀疏视角 k=2/3/4、dropout、count embedding、DLT-fallback |
| M5 跨域与 MPI 提交 | 09/30 – 10/06 | 1 周 | MPI 服务器提交、AIST++ 跨域验证 |
| M6 论文重写与图表 | 10/07 – 10/27 | 3 周 | 故事线、图表、表格、方法章节 |
| M7 内部审稿与修改 | 10/28 – 11/04 | 1 周 | 内审、补充实验、格式检查 |
| M8 截稿提交 | 11/05 – 11/14 | 1 周 | 最终版本、CMT 上传、视频/材料 |

**关键约束**

- A800 仅可使用 GPU 6/7；GPU 0–5 属于其他项目，严禁占用。
- 当前 A800 磁盘 `/mnt/nvme0n1p1` 约 **98% 满**，任何新实验前需先运行 `scripts/cleanup_a800_safe.sh` 干跑并清理。
- 截稿前至少保留 1 周缓冲，避免最后一刻 GPU/磁盘阻塞。

---

## 2. 里程碑详情

### M1 数据地基收尾（08/12 – 08/18，1 周）

| 项目 | 内容 |
|:---|:---|
| **目标** | 完成 `data/h36m_true_gt_v2/` 在 A800 的最终同步与验证；生成可复现的审计报告与 manifest。 |
| **负责人** | 主代理 + A800 子代理 |
| **任务清单** | 1. 确认 v2 `.npz` 在 A800 全部存在且 MD5/大小与本地一致。<br>2. 复现并记录 DLT baseline（25.67 mm）与 RANSAC/conf-DLT（26.47 mm）。<br>3. 更新 `configs/splits/h36m_true_gt_v2_standard.yaml` 与 `docs/results_true_gt_h36m.md`。<br>4. 运行 `scripts/cleanup_a800_safe.sh` 干跑，标记可清理文件。 |
| **验收标准** | - `data/h36m_true_gt_v2/` 下 S1/S5/S6/S7/S8/S9/S11 全部存在。<br>- `outputs/h36m_true_gt_v2_dlt_baseline.json` 与 `outputs/h36m_true_gt_v2_ransac_baseline.json` 存在且数值与本地一致（误差 < 0.5 mm）。<br>- `docs/results_true_gt_h36m.md` 的 label audit 表格更新。 |
| **风险点** | - A800 磁盘空间不足导致同步失败。<br>- v2 文件在传输中出现静默损坏，需校验。<br>- 外部项目占用 GPU 7，v2 验证可能只能排队 GPU 6。 |

---

### M2 基线重跑（08/19 – 09/01，2 周）

| 项目 | 内容 |
|:---|:---|
| **目标** | 在 `h36m_true_gt_v2` 上重跑所有核心基线，建立可信 leaderboard。 |
| **负责人** | A800 子代理（训练）+ 主代理（汇总） |
| **任务清单** | 1. 启动 **v25 v2 medium** 训练（true-GT v2， corrected validation）。<br>2. 启动 **v86 no-count-embedding** A800 中档训练（GPU 6/7 任一空闲时）。<br>3. 重跑 / 确认 Iskakov ICCV 2019（23.40 mm 基准）。<br>4. 汇总 DLT/RANSAC/Iskakov/v25/v86 结果到 `docs/results_true_gt_h36m.md`。 |
| **验收标准** | - v25 v2 训练完成并保存 checkpoint：`outputs/ablations/v25_true_gt_v2_medium_a800.pth`。<br>- v86 A800 训练完成：`outputs/ablations/v86_no_count_embedding_medium_a800.log` 完整。<br>- 所有基线 combined direct MPJPE 可复现，与现有数值差异 < 1 mm。<br>- `docs/results_true_gt_h36m.md` leaderboard 更新。 |
| **风险点** | - v25 v2 可能重现“验证不传递 view_mask” 问题，需提前检查 `motionflow_mv/eval/validation_loop.py`。<br>- v86 占用 GPU 时间，若 GPU 7 长期被外部占用，排队会拉长 M2。<br>- v25 v2 在 true-GT v2 上可能过拟合，需密切监控 val MPJPE 曲线。 |

---

### M3 SOTA 比较（09/02 – 09/15，2 周）

| 项目 | 内容 |
|:---|:---|
| **目标** | 完成 VoxelPose、MVPose、DLT 在 true-GT v2 上的公平对比。 |
| **负责人** | A800 子代理 + 主代理 |
| **任务清单** | 1. 跑通 VoxelPose 适配器（`scripts/sota_baselines/voxelpose_h36m_adapter.py`）。<br>2. 复现 / 确认 MVPose 结果（26.06 mm）。<br>3. 统一评估协议：同一 skeleton 子集、同一 PA 对齐、同一 PCK 阈值。<br>4. 生成 SOTA 对比表并写入 `docs/paper_draft_icra_cvpr_2027.md` 第 5 节。 |
| **验收标准** | - VoxelPose H36M true-GT 结果 JSON：`outputs/sota_baselines/voxelpose_h36m_true_gt_metrics.json`。<br>- MVPose / DLT / RANSAC / Iskakov / v25 结果在同一表格中可比较。<br>- 所有方法使用相同的 17-joint / 12-joint 子集定义。 |
| **风险点** | - VoxelPose 上游仓库环境可能与当前 `.venv` 冲突，需隔离 env 或 Docker。<br>- VoxelPose 在 true-GT 上可能极慢，需提前评估运行时间。<br>- 不同 SOTA 方法的 skeleton 映射不一致，需统一 adapter。 |

---

### M4 消融与鲁棒性曲线（09/16 – 09/29，2 周）

| 项目 | 内容 |
|:---|:---|
| **目标** | 产出系统的稀疏视角（k=2/3/4）鲁棒性曲线，明确 count embedding、dropout、DLT-fallback 的贡献。 |
| **负责人** | A800 子代理（实验）+ 主代理（分析） |
| **任务清单** | 1. 收集 v85 DLT-fallback 评估结果（PID 2269984 完成后）。<br>2. 完成 v86 no-fallback 与 DLT-fallback 变量视角评估。<br>3. 重跑 v25 / v81 / v82 DLT-fallback 作为基准。<br>4. 绘制 MPJPE@k 曲线（S9/S11/combined）。<br>5. 若 k<4 仍灾难性，设计并启动新消融：sparse-view head / stronger count conditioning / reweighted loss。 |
| **验收标准** | - 至少拥有以下完整曲线：v25、v81、v82、v85、v86（no-fallback + DLT-fallback）。<br>- 图表保存为 `docs/figures/sparse_view_mpjpek_cvpr2027.pdf` 与 `.png`。<br>- `docs/results_variable_views_curriculum.md` 更新。<br>- 若启动新消融，需有明确 go/no-go 决策点（09/22 前）。 |
| **风险点** | - v85 k<4 仍灾难性，可能需要额外架构改动，挤压 M5/M6 时间。<br>- v86 训练失败或结果无意义，需备用方案。<br>- 变量视角评估脚本多、输出路径分散，需严格命名规范。 |

---

### M5 跨域验证与 MPI 服务器提交（09/30 – 10/06，1 周）

| 项目 | 内容 |
|:---|:---|
| **目标** | 完成 MPI-INF-3DHP 官方服务器提交，补充 AIST++ / Shelf / Campus 跨域结果。 |
| **负责人** | 主代理 + A800 子代理 |
| **任务清单** | 1. 准备 MPI 服务器提交包（按 MPI 官方格式打包预测结果）。<br>2. 在 A800 上跑 MPI 测试集 learned 模型推理（v25 / v85 若可用）。<br>3. 确认 AIST++ → H36M 跨域结果（93.94 mm）已落盘并写入论文。<br>4. 更新 Shelf / Campus detected 结果到 `docs/results_true_gt_shelf_campus.md`。 |
| **验收标准** | - MPI 官方服务器返回提交成功确认邮件 / 结果页面。<br>- `outputs/mpi_submission_cvpr2027/` 包含完整提交文件与日志。<br>- 跨域表格（AIST++、MPI、Shelf/Campus）进入论文第 5 节。 |
| **风险点** | - MPI 服务器提交窗口可能有限，需提前确认截止日期。<br>- MPI 结果若与本地 DLT 差异过大，需排查格式错误。<br>- 跨域训练若未开始，可能无法产生 learned MPI 结果，只能提交 DLT baseline。 |

---

### M6 论文重写与图表（10/07 – 10/27，3 周）

| 项目 | 内容 |
|:---|:---|
| **目标** | 以“true-GT + 稀疏视角 + 跨域鲁棒性”为主线，完成 CVPR 2027 论文全文与图表。 |
| **负责人** | 主代理 + 子代理（绘图 / LaTeX） |
| **任务清单** | 1. 重写 Introduction & Related Work，强调 circular-label 问题与 true-GT pivot。<br>2. 完善 Method（第 3 节）：几何 triangulation + residual refinement + CamPE + GJR + dropout。<br>3. 生成/更新 Figure 1（honest leaderboard）、Figure 2（MPJPE@k 曲线）、Figure 3（跨域柱状图）。<br>4. 重写 Experiments（第 4–5 节），使用 v2 数字。<br>5. 准备 Supplementary：实现细节、额外消融、效率表。 |
| **验收标准** | - `docs/paper_draft_icra_cvpr_2027.md` 全文完成并反映 v2 数据。r>- 所有旧 circular-label 数字被替换或删除。r>- 图/表使用 `docs/figures/` 中最新版本。r>- 论文内部自洽：所有引用表格/图表可定位。 |
| **风险点** | - 若 M4 实验结果不理想，论文故事需重新包装，可能大幅返工。<br>- 绘图脚本依赖实验输出，实验未完成会导致图表空缺。<br>- 多人协作时 LaTeX / Markdown 版本冲突，建议使用 Git 分支管理。 |

---

### M7 内部审稿与修改（10/28 – 11/04，1 周）

| 项目 | 内容 |
|:---|:---|
| **目标** | 完成至少一轮内部审稿，修复漏洞，补充缺失实验。 |
| **负责人** | 主代理 + 外部审稿人（若可安排） |
| **任务清单** | 1. 组织内部 review：方法可解释性、实验充分性、故事线。<br>2. 检查所有数字与 `docs/results_true_gt_h36m.md` 一致。<br>3. 修复 reviewer 提出的主要问题（若涉及补充实验，限制在 3 天内可完成）。<br>4. 格式检查：CVPR 模板、页数、引用、图表分辨率。 |
| **验收标准** | - 内部审稿意见文档（`docs/proposals/cvpr2027_internal_review.md`）。<br>- 论文所有表格/图表与结果文件一致。<br>- 无明显方法漏洞或未被解释的异常数字。 |
| **风险点** | - 审稿人提出需要大实验的问题，可能无法在一周内完成。<br>- 最后一刻发现数字不一致，需追溯到原始 JSON。 |

---

### M8 截稿提交（11/05 – 11/14，1 周）

| 项目 | 内容 |
|:---|:---|
| **目标** | 在 CMT 提交最终论文、补充材料、视频。 |
| **负责人** | 主代理 |
| **任务清单** | 1. 生成 PDF / supplementary ZIP / 视频（若要求）。<br>2. 检查 CMT 元数据：作者、关键词、摘要。<br>3. 在 abstract deadline（预计 11/07）前注册标题和 abstract。<br>4. 在 full paper deadline（预计 11/14）前完成最终上传。 |
| **验收标准** | - CMT 显示提交成功，收到确认邮件。<br>- 最终 PDF 与 `docs/paper_draft_icra_cvpr_2027.md` 内容一致。<br>- 补充材料无遗漏。 |
| **风险点** | - 网络 / CMT 拥堵，需提前 24 小时完成上传。<br>- 视频渲染或文件大小超出限制。<br>- 最后一刻发现格式问题导致重新编译。 |

---

## 3. 依赖关系与关键路径

```
M1 数据地基
  │
  ▼
M2 基线重跑 ───────────────────┐
  │                             │
  ▼                             ▼
M3 SOTA 比较 ◄───────────────► M4 消融/鲁棒性
  │                             │
                               ▼
M5 MPI 提交 ◄───────────────── M4 结果
  │
  ▼
M6 论文重写
  │
  ▼
M7 内部审稿
  │
  ▼
M8 截稿提交
```

**关键路径：** M1 → M2 → M4 → M6 → M7 → M8。

---

## 4. 风险总览与应对

| 风险 | 影响 | 应对 |
|:---|:---|:---|
| A800 GPU 6/7 被外部项目长期占用 | 中 | 优先完成 M1/M2 中可在本地 RTX 4090 上验证的部分；A800 任务使用 nohup + 队列脚本。 |
| 磁盘 98% 满 | 高 | 每个新实验前运行 cleanup 干跑；删除已确认的 v83/v84 失败输出；保留 v85/v86 直到结果归档。 |
| v85/v86 k<4 仍灾难性 | 高 | M4 设置 09/22 go/no-go 决策点：若仍失败，论文转向“DLT-fallback + 学习化 k=4 增强”叙事。 |
| VoxelPose 跑不通 | 中 | 准备备选：仅对比 MVPose + Iskakov + DLT/RANSAC，并在论文中说明 VoxelPose 适配困难。 |
| MPI 提交窗口过期 | 中 | 在 M5 第一周即确认官方截止日期；若无法提交 learned 结果，提交 DLT baseline 作为诚实参考。 |
| 论文返工量大 | 中 | M6 预留 3 周，M7 预留 1 周；每周同步 `docs/paper_draft_icra_cvpr_2027.md` 进度。 |

---

## 5. 每周检查清单（主代理使用）

- [ ] **每周一：** 检查 A800 GPU 6/7 占用、磁盘空间、所有 running job 状态。
- [ ] **每周三：** 更新 `docs/results_true_gt_h36m.md` 与 `docs/results_true_gt_shelf_campus.md`。
- [ ] **每周五：** 更新本 milestone 文件，标记已完成/阻塞任务，必要时调整计划。
- [ ] **里程碑结束：** 完成验收标准并记录实际日期。

---

## 6. 相关文件索引

| 文件 | 用途 |
|:---|:---|
| `docs/results_true_gt_h36m.md` | H36M true-GT v2 排行榜 |
| `docs/results_true_gt_shelf_campus.md` | Shelf / Campus 检测结果 |
| `docs/paper_draft_icra_cvpr_2027.md` | 论文草稿 |
| `configs/splits/h36m_true_gt_v2_standard.yaml` | v2 标准协议 manifest |
| `scripts/cleanup_a800_safe.sh` | A800 安全清理脚本 |
| `scripts/run_h36m_true_gt_dlt_baseline.py` | DLT 基准 |
| `scripts/run_h36m_true_gt_ransac_baseline.py` | RANSAC 基准 |
| `scripts/sota_baselines/voxelpose_h36m_adapter.py` | VoxelPose 适配 |
| `scripts/sota_baselines/mvpose_h36m_adapter.py` | MVPose 适配 |

---

*Last updated: 2026-08-12*
