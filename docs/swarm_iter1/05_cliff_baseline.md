# CLIFF Baseline: Monocular 3D Human Pose

## TL;DR
CLIFF（Carrying Location Information in Full Frames）是 2022 年 ECCV 工作，输入单张 RGB，输出 SMPL 姿态、形状及弱透视相机参数。与传统仅对 crop 回归的方法不同，CLIFF 将人体在原始全图中的位置/尺度信息编码到网络输入中，从而得到更准确的全局 3D 位置，适合作为多视角融合的 monocular 前端。

## 关键结论
- 任务定位：单目 3D 人体姿态与形状估计，输出 SMPL 参数。
- 主要输出：
  - pred_rotmat：24 个关节相对旋转矩阵 [24,3,3]
  - pred_shape：SMPL 形状参数 [10]
  - pred_cam：弱透视相机 [s, tx, ty]
- 多视角用途：每个视角独立运行 CLIFF，获得 per-view 的 3D 人体参数，后续可做多视角融合/三角化/时序滤波。
- 限制：单帧方法，无显式时序约束；对遮挡与极端视角的鲁棒性取决于 backbone。

## 最小可运行示例
~~~bash
git clone https://github.com/paulchhuang/CLIFF.git
cd CLIFF
pip install -r requirements.txt
python demo.py --image sample.jpg --bbox 100 200 300 500 --model cliff_res50 --checkpoint checkpoints/cliff_res50_224x224.pth
~~~

## 参考链接
- 论文：CLIFF: Carrying Location Information in Full Frames into Human Pose and Shape Estimation (ECCV 2022)
- 代码仓库：https://github.com/paulchhuang/CLIFF
- 另一实现/参考：https://github.com/huifuhao/CLIFF
