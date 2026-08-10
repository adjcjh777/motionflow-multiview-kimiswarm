# Workflow Run Log

每次 Workflow 运行追加一条：日期、名称、agent 数、结论、后续动作。

---

## 2026-08-10 · recon-data-unblock-r2（完成）

- **目的**：P0 解阻塞侦察（5 agents：SemGCN 镜像审计、本地环境验证、获取渠道调研、A800 训练就绪、Iskakov 基线方案）。
- **结论**：
  1. **P0-1 可解（两条路，均已验证）**：
     - **A 路（零下载，最优）**：本地 `data/h36m_mirror/SemGCN/h36m.zip`（260 MB）内含官方 mocap `h36m/S{1,5,6,7,8,9,11}/MyPoses/3D_positions/*.h5`（210 个文件，32 关节、mm、世界系、骨长 rel-std 4e-7）。帧数与 2D 源 pkl **逐一精确匹配**（test 60/60、train 75/75）；关节映射 `[11,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]` 经 4 视角重投影验证 2.3–3.6 px RMSE。子动作拼接顺序 `['<Action> 1.h5', '<Action>.h5']` 由 2D/3D 速度相关 ≥0.7 证明。
     - **B 路（已下载备用）**：MHFormer GDrive `data_3d_h36m.npz`（182,626,216 B 精确匹配）已落在 `data/h36m_true_gt/`。
     - ⚠️ 发现 `experiments/prepare_h36m_true_gt.py` 的 `ACTION_NAMES` 对该 pkl **是错的**（7/12/13 错位、缺 16；正确表：07=Posing, 12=Photo, 13=Waiting, 14=Walking, 15=WalkDog, 16=WalkTogether）——任何接入路径都必须修正。
     - ⚠️ test-split 既有 canonical npz 相机数组退化（s_09 view3 重复 view0；s_11 仅 2 个独立相机），重建 test npz 时需逐行取 pkl `camera_name`。
     - 诚实注记：S1 Directions 上 circular DLT 标签与真 GT 仅差 14.57 mm——真 GT 的价值在协议正确性（无标签泄漏）与诚实基准，数字本身并非垃圾级修复。
  2. **P0-2 可解**：MPI 官方 `mpi_inf_3dhp_test_set.zip`（6,936,910,586 B）8-7 已下载完（md5 c65193d543324d85de992087ff1867fe），只是解压被打断；训练集 imageSequence 可按官方 get_dataset.sh 模式取。
  3. **本地环境**：`/d/anaconda3/python.exe` 是唯一即用即得的 4090 CUDA 解释器；WSL venv 缺 scipy/yaml/einops。
  4. **Iskakov 基线**：可复现——`triangulate_dlt_batched_lstsq`（可微分、支持逐视角逐关节权重）已存在；缺独立 baseline 训练脚本。
- **主循环直接动作（先于 Act workflow）**：代码 + manifest + 4 个 detected npz 打包（2 MB）scp 到 A800 `/mnt/nvme0n1p1/zhangzy/motionflow-mv-detected-long/`；v80 微 smoke 通过（epoch1 val 433.37 mm，与本地 smoke 一致，确认无环境漂移）；**v80 25-epoch 长跑已用 nohup 在 A800 GPU4,5 启动**（tmux 在该机器不可用；日志 `outputs/omniview_fusion_v80_shelf_campus_detected_long.log`）。
- **后续动作**：Act workflow——并行下载 H36M npz + MPI 测试集、实现 Iskakov baseline、跑双验收门。

---

## 2026-08-10 · recon-data-foundation（完成，被 r2 复核取代）

- **目的**：为 P0-1/P0-2 解阻塞做侦察。4 个并行 agent：
  1. pkl `joint3d_image` 真实性审计（能否本地恢复 H36M 真 3D）
  2. A800-D 只读侦察（GPU 清点、H36M mocap / MPI imageSequence 全盘搜索、远程仓库状态）
  3. H36M 真 GT / MPI 图像的可获取渠道调研（官方 + 社区镜像）
  4. 本地环境验证（GPU、单测、循环诊断复现、排行榜日志复核）
- **规模**：4 agents（medium 守则内）
- **模型**：子代理与主模型一致（不覆盖 model）
- **结论**：
  1. pkl 只有 `joint3d_image`（(u,v,z) 图像系），无法一致还原世界系 3D → 本地无法从 pkl 恢复真 GT，必须外部获取。
  2. A800-D 无 H36M mocap、无 MPI imageSequence；仓库镜像落后且脏，只能 scp 同步。
  3. 渠道：MHFormer GDrive `data_3d_h36m.npz`（~174 MB，HTTP 200）可用；MPI 官方站点存活、逐序列 zip 可下；fbaipublicfiles 3D 链接已 403。
  4. 本地环境验证通过（RTX 4090 + torch cu118，16 个 H36M 管线单测通过，循环诊断复现）。
- **后续动作**：已由主循环执行——下载 data_3d_h36m.npz、接入 `--data3d-npz`、过双验收门（P0-1 关闭）；MPI 训练图像下载 + detected-2D 生成进行中（issue #191）。细节见 r2 条目与 docs/results_true_gt_shelf_campus.md / results_iskakov_h36m_true_gt.md。

---

## 2026-08-10 · P0-1 h36m-true-gt-acquisition（完成，issue #194）

- **目的**：获取 H36M 真 3D GT 并接入项目管线（P0-1）。
- **规模**：单 track（workflow P0-1 分支）。
- **结论**：
  1. 主渠道成功：MHFormer GDrive `data_3d_h36m.npz`（182,626,216 B，精确匹配）下载至 `data/h36m_true_gt/data_3d_h36m.npz`；结构为 `positions_3d -> {"S{n}": {"<Action> [k]": (F,32,3) f32}}`，单位**米**（recon 预估的 mm 不对，已按重投影验证改为 ×1000）。
  2. `experiments/prepare_h36m_true_gt.py` 新增 `--data3d-npz`：按 pkl 帧数逐组精确匹配/拼接子动作（含排列搜索），17 关节选择 = 标准 H36M-17 `[0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]`（非 range(17)，已用重投影 3.9 px vs 262 px 证伪 range(17)）。
  3. 修复 test-split 相机退化：pkl `camera_name` 在 test 分割内逐行打乱，canonical 转换器改用固定 H36M 相机顺序（01..04 -> 54138969/55011271/58860488/60457274），重投影验证。
  4. 双验收门全过：7 个主体 canonical npz reproj RMSE 3.13–7.15 px（≤15 px 目标内），circularity direct MJE 27,762–37,320 mm（>>0）。DLT baseline：S9 29.54/21.81 mm（full）、S9 acts2+14 28.22 mm、S11 21.81 mm——在 15–30 mm 合理区间。
  5. 单测 21/21 通过（含新增 5 个 data3d 路径用例）。
- **产出**：`data/h36m_true_gt/` 下 8 个 canonical npz（S9/S11 全 test + S1/5/6/7/8 全 train）+ 日志 `s_*_gen.log`、`dlt_baseline_h36m*.json`、`diagnose_s9_s11_full.log`。
- **后续动作**：将 npz 写入训练 manifest；在修正协议上重跑 v25/v80 基线；A800 同步新 npz。

---

## 2026-08-10 · recon-data-foundation-r2（完成，4 agents）

- **目的**：对 P0 三条阻塞线做独立事实核查 + 下一步规划（本地镜像审计、A800 深搜、获取渠道、Iskakov 基线规划）。
- **结论**：
  1. **P0-1 已在本地解除**：`data/h36m_true_gt/data_3d_h36m.npz`（182,626,216 B）含全部 7 主体真 mocap 3D（522,774 帧）；`--data3d-npz` 适配器已消费，8 个 canonical npz 过双验收门（本 session 独立重验：S9 MJE 37,319.56 mm / S11 27,762.02 mm）。意外收获：`data/h36m_mirror/SemGCN/h36m.zip`（260 MB）内含相同真 mocap 的 `MyPoses/3D_positions/*.h5`（帧数与 data_3d_h36m.npz 精确一致），可作为全离线后备（需 ~5 行 .h5 adapter）。`h36m_corrected` 仍循环（0.0000 mm）已复核。
  2. **A800-D 深搜为负**：无 H36M 真 GT、无 MPI imageSequence；/mnt/nvme0n1 只是 nvme0n1p1 的 symlink，98% 已用（剩 73G）。GPU 0-3 被 vLLM 占满，4 当时已被占、5/6 空闲（≤2 GPU 规则内可用 5,6）；无 tmux；docker 容器 motionflow 等勿动。镜像仓库在 main@34d46eb 落后且有脏改动 → 只能 scp 同步，不可 git reset/pull。
  3. **获取渠道**：H36M 3D 首选保留现有文件；丢失时 MHFormer GDrive（id 1mAHq0YhO75frDkgUgebFQYnnPQOjUcr4，gdown 可无人值守，A800 无法访问 GDrive）；fbaipublicfiles 3D 链接 403、HF 无 3D 镜像。MPI 训练集图像官方站点存活、无需注册：16 个 `vnect_cameras.zip` 全 HTTP 200（共 6.34 GB）+ annot.mat + camera.calibration；test_set.zip 6.46 GB 已本地在。建议下载到本地 D:（178G 空闲）而非 A800。
  4. **Iskakov 基线已基本完成**：模块/训练器/结果文档/3 次运行均在（mixed 最优 combined direct 128.05 mm，超 conf-DLT +4.24 mm）。剩余：修正错误引用（"Iskakov, D., Kasneci, E." 系杜撰；正确为 Iskakov, Burkov, Lempitsky, Malkov, ICCV 2019, arXiv:1905.05754；另有 "Isakov" 拼写错误）、补单测、同步 leaderboard/issue。
- **主循环已执行动作**：引用修复已提交推送（c84db17）；MPI detected-2D 从 AVI zip 直接解码的新脚本 `scripts/generate_mpi_detected_2d_from_avi.py`（MediaPipe Tasks API，mediapipe≥1.0；顺序解码 0.2→2.4 frames/s，12× 加速；未映射关节 conf=0）已提交（bb7fb3f, b6d8a56），全量 16 序列生产运行中（4 workers, 384px）。
- **后续动作**：detected npz 过验收门 → DLT/v25/v57/v80 重跑 MPI 协议 → 更新 issue #191/#192/#193。

---

## 2026-08-10 · 主循环（非 workflow）· Iskakov 基线扩展 + v80 H36M 正则化 + MPI detected-2D 生产

- **目的**：把已解除的 P0-1 真 GT 协议跑通 baseline 与学习模型，同时推进 P0-2。
- **执行**（全部 GPU 纪律：A800 ≤2 张固定卡，本地 4090 单进程）：
  1. **Iskakov 基线补全**：修引用（Iskakov, Burkov, Lempitsky, Malkov, ICCV 2019, arXiv:1905.05754），新增 7 项单测全过（tests/test_iskakov_learnable_triangulation.py），trainer 支持 `--protocol h36m`（5 训练主体共享 1569 参数模型，S9/S11 逐主体 + 宏均值，SVD 参照用确定性 stride 采样）。同 seed 复跑：Campus-only 132.34/118.17 mm 与原 run 逐位一致；mixed 128.73 vs 原 128.05（+0.7 mm，源于为 H36M 协议改为逐步采样的 RNG 消耗差异）。
  2. **Iskakov 在 H36M 真 GT 上**：combined direct **23.38** mm（+2.49 vs conf-DLT 25.87，+5.81 vs unweighted 29.19）；S9 27.13 / S11 19.64 mm，落在 15–30 mm 合理带。docs/results_iskakov_h36m_true_gt.md。
  3. **v80 在 H36M 真 GT（A800 GPUs 4,5）**：lr1e-3/wd0 → epoch2 后过拟（best 65.28 → 501 mm）；lr5e-4/wd1e-4 → epoch2 新 best 39.70 但 epoch3 发散（168 mm）；已杀进程、保存 epoch-2 最优 ckpt，改 lr2e-4/wd5e-5/patience2 重跑（运行中）。DLT 锚点 S9 29.54 / S11 21.81 mm。
  4. **Shelf/Campus 长跑结论入库**：v80 best 276.49 mm（epoch7）/v57 best 306.45 mm（epoch4），均随后过拟至 784/834 mm；更长训练不足以在 ~1k 帧上逼近 DLT（122.37 root）。issue #193 置 DONE。
  5. **P0-2**：新脚本从 AVI zip 直接解码（12× 提速），15 序列生产运行中（4 workers）；测试集 zip 已本地、TS 转换脚本已存在。issue #191 更新中。
  6. **本地 v25 H36M medium**（4090 单进程）：8 epoch、1024 samples/epoch，验证 v25 在真 GT 上量级合理（无 NaN）——对应 CLAUDE.md 完成标准"DLT 与 v25 得 15–30 mm"。
- **后续动作**：等 MPI detected npz 完成 → 双验收门 → DLT + Iskakov + v25/v57/v80 重跑 MPI 协议 → 更新 #191；等 A800 v80 reg run 收敛 → 记录并比对 DLT 锚点。
