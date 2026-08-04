# Fusion Model Training Loop: Supervised 3D Target

## TL;DR
针对 MotionFlow 多视角扩展，采用**冻结/预提取的 per-view 2D 关节点 + 置信度**作为输入，以**GT 3D 关节点**为监督目标训练一个轻量级 FusionHead。训练循环只包含 3D L2/MPJPE loss 与可选的 2D 重投影一致性 loss；验证阶段计算 MPJPE / PA-MPJPE，不重新训练 2D 检测器。

## 关键结论

1. **仅训练 FusionHead**：保持 MotionFlow 等 2D/单目基线模型冻结，将多视角融合建模为 2D
