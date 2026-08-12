# 3DPW 数据集调研

## TL;DR

3DPW（3D Poses in the Wild）是 ECCV 2018 发布的野外单目视频人体数据集，包含 60 段视频、2D/3D 标注、每帧相机位姿以及 **SMPL 参数级 Ground Truth**。它适合作为 MotionFlow 单目 baseline 的验证集或跨域测试，但**不是多视角数据集**，不能作为本项目的核心多视角训练/测试数据。

## 1. SMPL Ground Truth

3DPW 的核心价值在于提供 SMPL 模型参数真值：
- `pose`: 72-D (24 joints × 3) 轴角
- `betas`: 10-D PCA 形状参数
- `trans`: 全局平移
- 同时提供 3D 关节点位置、每帧相机位姿 `campose_para`

数据组织为 `.pkl` 序列文件，官方按 train/val/test 划分。

## 2. Video & Camera

- 60 段野外视频，由移动手持设备拍摄
- 提供逐帧相机位姿（视觉-惯性里程计）
- 图像是单目序列，**非同步多视角**

## 3. 对本项目的适用性

| 维度 | 结论 |
|------|------|
| SMPL GT | 有，可评估 3D pose/shape |
| 视频 | 有连续视频，可测试时序方法 |
| 多视角 | 否，仅单目移动相机 |
| 规模 | 仅 60 序列，较小 |
| 许可 | 需签署协议并单独下载 SMPL 模型 |

### Pros
- 稀有的 in-the-wild SMPL 真值数据集
- 含相机参数，可验证重投影、轨迹等任务
- 被 VIBE/SPIN/ROMP 等广泛采用，便于对比

### Cons
- **非多视角**：本项目核心是多视角融合，3DPW 无法直接提供多机位同步视频
- 数据量小，不适合训练大规模融合网络
- 下载和 SMPL 模型均需授权

## 4. 最小可运行示例

```python
import pickle

seq_path = "data/3DPW/sequenceFiles/test/courtyard_basketball_00.pkl"
with open(seq_path, "rb") as f:
    seq = pickle.load(f, encoding="latin1")

print(seq.keys())
# 'poses', 'betas', 'trans', 'cam_poses', 'campose_para', ...
print(seq["poses"].shape)  # (n_frames, 72)
```

## 5. 参考链接

- 3DPW 官网与下载：https://virtualhumans.mpi-inf.mpg.de/3DPW/
- 论文：von Marcard et al., "Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera", ECCV 2018
- mmhuman3d 预处理说明：https://github.com/open-mmlab/mmhuman3d/blob/main/docs/preprocess_dataset.md
