# WHAM 单目基线调研

## TL;DR

WHAM（World-grounded Human Motion）是一个端到端的单目视频人体运动估计器，输出世界坐标系下的 SMPL 动作和相机轨迹，适合作为 MotionFlow 多视角扩展的替代/参考基线。

## 关键信息

| 项目 | 内容 |
|---|---|
| 论文 | Shin et al. *“WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion.”* CVPR 2024. arXiv:2312.07531 |
| 仓库 | https://github.com/yohanshin/WHAM |
| 许可证 | MIT |
| 输入 | 单目 RGB 视频（可配可选的 SLAM 轨迹） |
| 输出 | SMPL `pose` / `shape`、3D 关节点、相机轨迹、2D 投影 |
| 依赖 | PyTorch, torchvision, SMPL/SMPL-X, OpenCV, PyTorch3D 等 |

## 最小可运行示例

```bash
git clone https://github.com/yohanshin/WHAM.git
cd WHAM
# 安装依赖并注册下载 SMPL 模型（见 docs/INSTALL.md）
bash fetch_demo_data.sh

# 仅估计相机坐标系运动（无需 SLAM）
python demo.py --video examples/IMG_9732.mov --visualize --estimate_local_only
```

## Python API 片段

```python
from wham.predictor import WHAM
import torch

model = WHAM().cuda()
# video: (T, 3, H, W) torch.Tensor, 输出包含 pose, shape, joints3d
out = model(video)
```

## 与多视角融合的关系

- 每个视角可独立运行 WHAM，得到各视角的 SMPL/3D 关节点。
- 若 WHAM 输出世界坐标，则不同视角的 3D 结果理论上应已对齐；实际中各视角相机轨迹独立，存在漂移。
- 更实际的做法是：用 WHAM 提取每视角的 2D 关键点 + 置信度，再通过多视角融合得到统一 3D 骨架。

## 参考

- https://github.com/yohanshin/WHAM
- https://arxiv.org/abs/2312.07531
