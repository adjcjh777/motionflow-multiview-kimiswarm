# A800 GPU 使用策略

> 生效日期：2026-08-12
> 约束级别：Goal-level / hard

## 规则

MotionFlow-MultiView 项目仅在 A800 的 **GPU 6 和 GPU 7** 上运行训练/评估任务。

- **GPU 0–5**：保留给其他项目（VLLM 等），本项目禁止使用。
- **GPU 6–7**：本项目专用。所有 `CUDA_VISIBLE_DEVICES` 必须设置为 `6` 或 `7`。
- 禁止使用 `CUDA_VISIBLE_DEVICES=4` 或 `5` 的脚本/命令。

## 影响范围

- 训练脚本：`scripts/run_*_a800_gpu*.sh`
- 评估脚本：`scripts/eval_*_a800_gpu*.sh`、`experiments/eval_variable_views.py`
- 监控 cron：每 30 分钟检查 GPU 6/7 状态
- 手动调试：本地 smoke 除外，仍可在 RTX 4090 上运行

## 违规处理

若发现 MotionFlow 进程占用 GPU 0–5：
1. 记录违规 PID 和命令行。
2. 在本次会话中报告给用户/主代理，不自动 kill（避免误伤）。
3. 若该进程属于本项目且用户/主代理已批准迁移，则 kill 后在 GPU 6/7 重新启动。

## 当前占用（示例）

| GPU | 任务 | 状态 |
|-----|------|------|
| 6   | v82 variable-view DLT-fallback eval | RUNNING |
| 7   | v85 random view dropout medium      | PLANNED |
| 6/7 | v81 / v25 DLT-fallback eval          | PLANNED |

## 相关脚本

- `scripts/wait_for_a800_gpu.sh` — 仅返回 GPU 6 或 7 中的空闲卡
- `scripts/run_v85_random_view_dropout_medium_a800_gpu7.sh` — v85 默认 GPU 7
