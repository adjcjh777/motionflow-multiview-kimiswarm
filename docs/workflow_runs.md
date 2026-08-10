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

## 2026-08-10 · recon-data-foundation（运行中）

- **目的**：为 P0-1/P0-2 解阻塞做侦察。4 个并行 agent：
  1. pkl `joint3d_image` 真实性审计（能否本地恢复 H36M 真 3D）
  2. A800-D 只读侦察（GPU 清点、H36M mocap / MPI imageSequence 全盘搜索、远程仓库状态）
  3. H36M 真 GT / MPI 图像的可获取渠道调研（官方 + 社区镜像）
  4. 本地环境验证（GPU、单测、循环诊断复现、排行榜日志复核）
- **规模**：4 agents（medium 守则内）
- **模型**：子代理与主模型一致（不覆盖 model）
- **结论**：（待补）
- **后续动作**：（待补）
