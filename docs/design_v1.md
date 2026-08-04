# MotionFlow Multi-View v1 设计方案

> 第一轮自进化：先跑通“多视角 2D → 物理空间 3D”的最小闭环，再逐步加入可学习融合。

## 1. 问题与目标

现有 MotionFlow  pipeline 只能处理单目视频。真实物理世界更常见的是**多视角同步视频**。本阶段目标：在保留单目 MotionFlow 能力的前提下，增加一个轻量级多视角融合模块，将同一动作在不同视角下的 2D/3D 人体关键点对齐到统一的物理（世界）坐标系。

**核心问题**：给定 N 个已标定相机的同步视频，如何得到一致、鲁棒的世界坐标 3D 人体骨架？

## 2. 本轮设计决策（禁止过度设计）

| 模块 | 决策 | 理由 |
|---|---|---|
| 单目 baseline | 将 MotionFlow 抽象为 `BasePoseEstimator` 接口 | 仓库中 MotionFlow 实现未公开/无法直接访问；接口化后可替换为 WHAM/VIBE/4D-Humans |
| 输出格式 | 每视角输出 `(J, 2)` 2D 关键点 + `(J,)` 置信度 | 2D 检测比 3D SMPL 更通用、计算量更小 |
| 融合策略 | 第一版：**置信度加权 DLT 三角化** | 无训练、确定性强、遮挡鲁棒，先验证端到端流程 |
| 相机标定 | 外部提供 `K, R, t`，构造成 `Camera` 对象 | Shelf/Campus/H36M 等数据集均已标定 |
| 数据集 | **Shelf / Campus** | 规模小、已标定、有 3D GT，适合 4090/A800 快速迭代 |
| 评价指标 | MPJPE、PA-MPJPE、PCK@0.05m | 标准 3D 姿态指标 |
| 训练 | 本轮不训练 | 先建立无训练 baseline，第二轮再加入可学习融合 |

## 3. 系统架构

```
同步多视角视频 / 帧序列
        │
        ▼
┌─────────────────────┐
│  per-view 2D detector │  ← MotionFlow / WHAM / VIBE / 4D-Humans 的 2D 输出
│  (black-box adapter)  │
└─────────────────────┘
        │
        ▼
  {view_0: (J,2)+conf, ..., view_V: (J,2)+conf}
        │
        ▼
─────────────────────┐
│   Camera model      │  ← K, R, t 投影矩阵
│   + Calibration     │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Confidence-Weighted│
│ DLT Triangulation │
└─────────────────────┘
        │
        ▼
   world 3D skeleton (J, 3)
```

## 4. 模块定义

### 4.1 抽象接口 `motionflow_mv/baseline/base.py`

```python
from abc import ABC, abstractmethod
import numpy as np

class BasePoseEstimator(ABC):
    @abstractmethod
    def extract(self, video_path: str) -> dict:
        """
        Returns:
            dict: {
                "keypoints_2d": np.ndarray,  # (T, J, 2)
                "confidence": np.ndarray,   # (T, J)
            }
        """
        pass
```

### 4.2 相机模型 `motionflow_mv/calibration/camera.py`

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Camera:
    K: np.ndarray   # (3, 3)
    R: np.ndarray   # (3, 3)
    t: np.ndarray   # (3,)

    @property
    def projection_matrix(self) -> np.ndarray:
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return self.K @ Rt  # (3, 4)
```

### 4.3 融合 `motionflow_mv/fusion/triangulation.py`

实现：
- `triangulate_dlt(points_2d, proj_matrices)`：基础 DLT。
- `triangulate_confidence_weighted(points_2d, proj_matrices, confidences)`：加权 DLT。

## 5. 实验计划

1. **数据**：Shelf / Campus 数据集（优先 Shelf，5 个视角）。
2. **2D 检测器**：使用现成的 2D 人体姿态估计器（如 RTMPose / OpenPose / MMPose）生成每视角 2D 关键点 + 置信度。若时间紧，可先用数据集提供的 2D GT 验证三角化模块正确性。
3. **流程**：
   - 读取相机参数
   - 读取/生成每视角 2D 关键点
   - 置信度加权 DLT 三角化
   - 与 3D GT 计算 MPJPE / PA-MPJPE
4. **对照**：单视角最佳结果（取某一视角的 2D 点反投影/三角化） vs. 多视角融合结果。

## 6. 下一轮（v2）方向

- 引入轻量 `ViewAttentionFusion`（参考 `docs/swarm_iter1/08_attention_fusion.md`），用可学习注意力替代固定置信度加权。
- 加入骨骼长度约束 / 时序平滑 loss，提升物理合理性。
- 在 A800-D 上训练 fusion head，4090 上验证。

## 7. 反馈闭环

每一轮结束后：
1. 在 GitHub Issue 中记录实验结果。
2. 收集团队/导师/审稿反馈。
3. 根据反馈调整下一轮设计（v2 融合模型、数据增强、训练策略等）。
4. 开启下一轮 PR。

## 8. 参考

- `docs/swarm_iter1/01_motionflow_baseline.md`
- `docs/swarm_iter1/06_dlt_triangulation.md`
- `docs/swarm_iter1/07_confidence_fusion.md`
- `docs/swarm_iter1/08_attention_fusion.md`
- `docs/swarm_iter1/13_shelf_campus_dataset.md`
- `docs/swarm_iter1/16_evaluation_metrics.md`
