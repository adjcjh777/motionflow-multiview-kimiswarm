# MotionFlow-MultiView — Claude 项目指南

> **当前状态：DATA FOUNDATION REPAIR → CVPR 2027（~2026-11）**
> 本文件由 `AGENTS.md` + `docs/handoff_qwen3.8max.md` + `docs/roadmap_cvpr2027.md` 汇总而成，是所有 agent 的第一阅读入口。

## 1. 项目一句话

多视角 3D 人体姿态估计研究项目（MotionFlow 的多视角扩展），目标是发表 CVPR 2027 论文。论文卖点已从"绝对 MPJPE 记录"转向 **稀疏视角 / 跨域鲁棒性**。

## 2. 核心事实（必读，违反即错误）

1. **H36M 标签是循环的**：`data/h36m_hf/*_multiview.npz` 与 `data/webbridge/h36m_corrected/`、`h36m_meters/` 的 `joints_3d` 都是 `DLT(points_2d, cameras)` 三角化的结果（direct MJE = 0.0000 mm）。v25–v79 在 H36M 上的所有数字**全部不可用于模型选择**。
2. **循环来源**：`motionflow_mv/data/webbridge_loader.py:182` 的 `_triangulate_joints` 把输入 2D 三角化后存成 3D 标签。永远不要把该函数的输出当作 GT。
3. **可信的 true-GT 数据**：
  - Shelf/Campus：`data/webbridge/shelf_campus_detected/`（真实检测 2D + 真实标注 3D，已验证非循环）。**Campus（3 视角）是主稀疏视角基准**；Shelf 标定粗糙（reproj RMSE ~53.7 px），引用时必须带 caveat。
  - MPI-INF-3DHP：3D 标签是真的（`univ_annot3`），但当前 2D 输入是 GT 投影；标准协议需要真实检测 2D（`imageSequence/` 在训练集上缺失，只有 `annot.mat` + `camera.calibration`；测试集 TS1 有 imageSequence）。
4. **H36M 真 GT 缺失**：官方 `PosesD3_Positions` 在本地与 A800-D 上均未找到。转换管线已就绪并单测通过：`experiments/prepare_h36m_true_gt.py` + `tests/test_h36m_true_gt_pipeline.py` + `scripts/fetch_h36m_true_gt.py`，等数据落到 `data/h36m_true_gt/`。
5. **不要引用已作废的数字**：如 "~17 mm H36M v25"、"~10.2/10.9 mm Shelf/Campus PA-MPJPE" 都是循环协议产物。



## 3. 当前排行榜（true-GT Shelf/Campus，2026-08-10，3-epoch smoke）


| 方法                                | Best val MPJPE (mm) |
| --------------------------------- | ------------------- |
| DLT root-aligned                  | **122.37**          |
| DLT direct MJE                    | **134.43**          |
| v80（视图可靠性加权，965k 参数）              | 408.58              |
| v57（domain-conditional PSC，1009k） | 424.63              |
| v25（geometry fusion，2732k）        | 430.67              |


学习模型排序：v80 > v57 > v25；全部远未收敛。详见 `docs/results_true_gt_shelf_campus.md`。

## 4. P0 阻塞（解除后才能做模型选择）


| ID   | 阻塞              | 状态                                                             | 解除步骤                                                                                           |
| ---- | --------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| P0-1 | H36M 真 3D GT 缺失 | 管线就绪，缺数据                                                       | 获取 `PosesD3_Positions` → `data/h36m_true_gt/` → `experiments/prepare_h36m_true_gt.py` → 通过双验收门 |
| P0-2 | MPI 真实检测 2D 缺失  | 脚本就绪（`scripts/generate_mpi_detected_2d.py`），缺 `imageSequence/` | 获取训练集图像 → 跑真实检测器 → 重跑 DLT/v25/v57/v80                                                          |
| P0-3 | 标准 SOTA 基线未复现   | 未开始                                                            | 统一协议下复现 Iskakov ICCV 2019 / RANSAC-DLT                                                         |


**验收门（任何新 true-GT npz 必须双过）**：

1. `python scripts/diagnose_circular_labels.py <npz>` → direct MJE 必须 >> 0 mm
2. `python scripts/check_true_gt_reprojection.py <npz>` → reproj RMSE ≤ ~15 px



## 5. 资源与 GPU 规则（2026-08-10 用户授权版）

- **A800-D（ssh** `a800-D`**）**：允许用于实验验证，**固定至多 2 张 GPU**（用 `CUDA_VISIBLE_DEVICES` 锁定固定编号，先 `nvidia-smi` 看占用再选）。仓库镜像：`/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`。数据目录 `/mnt/nvme0n1/zhangzy/projects`。禁止重启 Docker；tmux 会话先 `tmux ls` 检查再决定。（注：本条已覆盖旧 handoff 的"A800 只读"约束；数据目录仍只读，勿写入他人目录。）
- **本地 WSL + RTX 4090**：诊断 / smoke / 单卡训练验证。**不要同时跑多个训练进程**。
- **Dynamic Workflow**：一次最多 100 个子代理；独立任务优先并行；用 dynamic workflow 推进。
- **数据纪律**：真 GT / 真实检测 2D 就位前，smoke 数字只用于稳定性检查，不用于模型选择。



### 5.1 本机 shell 环境（已验证 2026-08-10，bash 子进程必读）

- Git Bash 的 PATH 默认是坏的（coreutils 不在路径上）。**每条 Bash 命令开头先执行**：
`export PATH="/mingw64/bin:/usr/bin:/bin:/c/Windows/System32:/c/Program Files/Git/cmd:$PATH"`
- Python：`/d/anaconda3/python.exe`（3.13.9，torch 2.7.1+cu118，CUDA 可用，单卡 RTX 4090 = GPU 0）。
- `nvidia-smi`：`/c/Windows/System32/nvidia-smi.exe`。
- WSL：`/c/Windows/System32/wsl.exe`，发行版 Ubuntu（Running）。
- GitHub：无 `gh` CLI；用 `curl` + GitHub REST API，token 从 `git config --get remote.origin.url` 提取（**绝不在命令行明文写 token**，会被安全分类器拦截）。
- A800-D 已验证可 ssh（BatchMode 可用）：主机 `myllm-002`，8× A800-SXM4-80GB，当前无 tmux 会话（训练已停）。



## 6. 关键文件地图


| 用途                              | 路径                                                                      |
| ------------------------------- | ----------------------------------------------------------------------- |
| 训练入口（mixed loader，domain 数自动设置） | `experiments/train_omniview_fusion_v5_webbridge_multi.py`               |
| Shelf/Campus 非循环 manifest       | `configs/splits/shelf_campus_detected_smoke.yaml`                       |
| 循环标签诊断                          | `scripts/diagnose_circular_labels.py`                                   |
| true-GT 重投影审计                   | `scripts/check_true_gt_reprojection.py`                                 |
| Shelf/Campus 重建脚本               | `experiments/build_shelf_campus_canonical_from_detection.py`            |
| H36M true-GT 提取                 | `experiments/prepare_h36m_true_gt.py` / `scripts/fetch_h36m_true_gt.py` |
| MPI 检测 2D 生成                    | `scripts/generate_mpi_detected_2d.py`                                   |
| MPI DLT baseline                | `scripts/run_mpi_dlt_baseline.py`                                       |
| MPJPE@k 评测                      | `scripts/run_mpjpe_at_k_benchmark.py`                                   |
| 排行榜工具                           | `scripts/leaderboard.py` / `scripts/aggregate_run_results.py`           |
| 结果文档                            | `docs/results_true_gt_shelf_campus.md` / `docs/roadmap_cvpr2027.md`     |
| 交接文档                            | `docs/handoff_qwen3.8max.md`                                            |
| 数据地基阻塞详情                        | `docs/data_foundation_blocker.md`                                       |




## 7. 常用命令

```bash
# 诊断任意 .npz 是否循环
python scripts/diagnose_circular_labels.py data/webbridge/shelf_campus_detected/shelf_seq1_val_detected_m.npz

# true-GT 重投影一致性审计
python scripts/check_true_gt_reprojection.py data/webbridge/shelf_campus_detected/*.npz --threshold 25

# v25 true-GT smoke（本地 4090）
bash scripts/run_v25_shelf_campus_noncircular_smoke_local_4090.sh

# A800 状态查看（先读后写）
ssh a800-D "nvidia-smi && tmux ls"

# 单测 H36M true-GT 管线
python -m pytest tests/test_h36m_true_gt_pipeline.py -q
```



## 8. GitHub 协作规范

- 远程仓库：`adjcjh777/motionflow-multiview-kimiswarm`。**及时更新 issue 与 PR**：每完成一个可交付物，更新对应 issue（状态前缀 `[RUNNING]/[BLOCKED]/[READY]/[DONE]`），新任务先开 issue 再动手。
- 标签体系：`P0-blocker`（里程碑前必须解除）、`P1-next`、`P2-nice`、`experiment`、`ablation`、`data`、`paper`、`infra`、`bug`。
- 分支命名沿用 `feat/iter-next-*` / `swarm/v*` / `fix/*` 惯例；PR 标题带版本号（如 `[v80] ...`）。



## 9. 论文写作红线

- **禁止编造引用**：是 Iskakov et al.（Learnable Triangulation, ICCV 2019），不是 "Iskandar"；"Ray-attention" 不存在，不要引用。
- 卖点：稀疏视角鲁棒性（MPJPE@k 曲线）、跨域泛化、标定鲁棒融合、诚实非循环基准。
- 所有数字必须来自非循环协议且附运行日志路径（见 `docs/results_true_gt_shelf_campus.md` 的 Evidence 表格格式）。



## 10. Dynamic Workflow（多代理编排规范）

本项目使用 Claude Code 的 **Workflow 工具** 推进探索任务（用户明确授权）。执行模式：

1. **混合式推进**：主循环先做轻量侦察（读 handoff / git log / issue 列表）确定工作清单，再用 Workflow 对工作清单做流水线/并行 fan-out。
2. **阶段化**：`Recon（侦察）→ Decide（决策合并）→ Act（执行）→ Verify（验证）→ Synthesize（汇总成 docs + issue/PR）`。阶段之间是否需要 barrier 取决于是否真的需要跨条目上下文：去重、汇总、"0 结果则提前退出"用 barrier；否则一律用 `pipeline()` 不等待。
3. **规模守则**：默认 medium（<15 agents）；独立任务并行，GPU 训练类任务串行（4090 单进程、A800 ≤2 GPU 且编号固定）。
4. **结构化返回**：每个 agent 用 schema 返回可机读结果（JSON），主循环据此决定下一轮分支——这就是 "dynamic"：工作流按发现的结果动态分叉（如发现 A800 有真 GT → 走拷贝分支；没有 → 走 pivot 分支）。
5. **验证门**：任何数据类产出必须过双验收门（§4）；任何模型结论必须附日志路径。Workflow 的最后一个阶段固定为"对抗验证"：让独立 agent 尝试反驳前一阶段的结论。
6. **失败处理**：agent 返回 null 时过滤掉并记录；Bash 安全分类器暂时不可用时重试或降级为只读操作，不要伪造结果。
7. **模型一致性**：所有子代理**维持与主模型一致**——Workflow `agent()` 调用与 Agent 工具一律不传 `model` 覆盖（继承主循环模型），保证推理质量统一。

已跑过的 workflow 与结论记录在 `docs/workflow_runs.md`（每次运行追加一条：日期、名称、agent 数、结论、后续动作）。

## 11. 自迭代研究方法（参考 Qwen3.8 self-evolution）

- **闭环**：action → 可验证反馈（诊断脚本/单测/排行榜）→ 诊断 → 重试。每次实验必须有可复现的验证命令。
- **多源信号**：内部（单测、DLT 诊断）+ 外部（文献标准协议、社区实现）结合。
- **证据门控**：小 smoke 通过才上 medium，medium 通过才上 full/A800；每个决定记录在 AGENTS.md 表格。
- **几何即验证器**：重投影误差、骨长一致性、对称性是比 loss 更可靠的反馈信号。



## 11. 完成标准（/goal 对应）

- [ ] H36M 真 GT 重建标准协议 .npz，DLT 与 v25 得 15–30 mm MPJPE
- [ ] MPI 真实检测 2D 生成 .npz，重跑 DLT/v25/v80/v57 排行榜
- [ ] 完成 Iskakov ICCV 2019 baseline 复现
- [ ] 重写论文卖点与引用
- [ ] 及时同步 GitHub issue/PR