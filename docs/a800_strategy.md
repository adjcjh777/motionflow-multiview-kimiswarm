# A800 大型实验策略

> 更新时间：2026-08-11
> 策略：本地 WSL（RTX 4090）只做快速 smoke/验证；大型训练任务迁移到 A800。

## 背景

用户明确指示："本地 wsl 做快速验证，大实验在 a800 上做"。因此：

- 本地 RTX 4090 不再运行超过 30 分钟的大型训练任务。
- A800 8× A800-SXM4-80GB 用于 medium/long 训练、消融矩阵、SOTA 基线。
- 本地保留：
  - smoke 验证（≤2 epoch，≤200 样本）
  - 数据/脚本准备
  - 文档更新
  - 代码/配置调试

## A800 现状

- 主机：`a800-D`
- 仓库：`/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`
- GPU：8× A800-SXM4-80GB，当前大多空闲
- 缺失：
  - `data/h36m_true_gt/`（真 GT `.npz`）
  - `data/h36m_hf/h36m_sh_conf_cam_source_final.pkl.zip`（源 pkl）
  - 本仓库最新 ablation 脚本（在本地 working tree）

## 迁移步骤

1. 将 `data/h36m_hf/h36m_sh_conf_cam_source_final.pkl.zip` 复制到 A800。
2. 在 A800 生成 `data/h36m_true_gt/` 真 GT `.npz`。
3. 将本地最新 ablation 脚本/配置同步到 A800（优先 scp 关键文件，避免 git 冲突）。
4. 在 A800 启动 medium/long 训练任务（tmux/nohup）。
5. 本地只保留 smoke 和验证。

## 当前本地状态

- 已停止本地 `v25_true_gt_baseline_fix` ablation（PID 16924），释放 RTX 4090。
- v25/v80/v57 medium 已跑完，结果在 `docs/results_true_gt_h36m.md`。
- 后续 ablation/大实验全部转到 A800。

## 下一步

- 完成数据/脚本迁移。
- 在 A800 启动 `v25_true_gt_baseline_fix` ablation。
- 在本地用 RTX 4090 验证 smoke/配置是否可运行。
