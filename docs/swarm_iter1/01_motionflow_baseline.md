# MotionFlow 基线调研

## TL;DR

公开仓库中**未找到**与 README 描述完全匹配的 “monocular video → human motion” 的 **MotionFlow** 代码库。最接近的公开项目是 `MohsenZand/MotionFlow`（Zand et al., TPAMI 2023），但它输入的是动作捕捉/序列数据，而非单目视频。建议将 MotionFlow 视为一个黑盒单目估计器，或改用公开的 WHAM / VIBE / 4D-Humans 作为可复现 baseline。

## 1. 公开候选：`MohsenZand/MotionFlow`

| 项目 | 信息 |
|---|---|
| 论文 | Zand, Etemad, Greenspan. *“Flow-based Spatio-Temporal Structured Prediction of Motion Dynamics.”* IEEE TPAMI, 2023. arXiv:2104.04391 |
| 仓库 | https://github.com/MohsenZand/MotionFlow |
| 许可证 | GPL-3.0 |
| 任务 | 运动预测、轨迹预测、时间序列预测、二值分割 |
| 输入 | 预处理后的 mocap/序列数据或分割图像，**不是原始单目视频** |
| 输出 | 预测的运动序列或分割掩码 |
| 架构 | 条件归一化流（Conditional Normalizing Flows）+ 掩码卷积；确定性/随机表示结合；自回归时序条件 |
| 依赖 | PyTorch；Python 3.8.8；CUDA 11.2（在 Ubuntu 18.04 + Titan RTX 测试） |

## 2. 推荐公开单目 baseline（视频输入）

### WHAM（CVPR 2023）
- 仓库：https://github.com/yohanshin/WHAM
- 许可证：MIT
- 输入：单目 RGB 视频
- 输出：SMPL 姿态参数、相机轨迹、世界坐标系 3D 人体运动
- 快速体验：
  ```bash
  git clone https://github.com/yohanshin/WHAM.git
  cd WHAM
  # 按 docs/INSTALL.md 安装依赖并下载 SMPL 模型
  python demo.py --video input.mp4 --visualize --estimate_local_only
  ```

### 4D-Humans / HMR2.0（CVPR 2023）
- 仓库：https://github.com/shubham-goel/4D-Humans
- 许可证：MIT
- 输入：单图 / 视频帧
- 输出：SMPL 参数、3D 关节点、2D 投影、跟踪 ID
- 可作为单帧/单目估计器集成到多视角融合流程中。

## 3. 建议

1. 如果 “MotionFlow” 是团队内部仓库，请在 Issue #1 或 README 中补全地址与分支。  
2. 若公开仓库即目标，建议以 **WHAM** 或 **4D-Humans** 作为黑盒单目估计器，先验证多视角融合逻辑。  
3. 集成多视角融合模块前，需确定 baseline 输出格式（2D 关键点、3D 关节点、SMPL `theta/beta` 或相机参数），不同输出决定下游 DLT/加权融合的具体实现。
