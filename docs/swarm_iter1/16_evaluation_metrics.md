# 16 人体姿态评估指标速查

## TL;DR
- **MPJPE**：无对齐的逐关节平均欧氏距离，最常用。
- **PA-MPJPE**：先 Procrustes 对齐再算 MPJPE，衡量“形状/姿态”精度。
- **PCK / AUC**：以关节到真值距离占参照长度的比例为阈值，统计正确率。
- **Procrustes**：仅是一种刚/相似对齐方式，常作为 PA-MPJPE 的前置步骤。

## 关键结论

### 1. MPJPE（Mean Per Joint Position Error）
\[ \text{MPJPE} = \frac{1}{N}\sum_{i=1}^{N} \|p_i - \hat{p}_i\|_2 \]
- 直接对 3D 坐标求 L2 距离。
- **何时用**：需要绝对位置精度的场景；缺点是受全局平移/旋转/尺度影响大。

### 2. PA-MPJPE（Procrustes Aligned MPJPE）
- 先通过 Procrustes 分析将预测与真值进行平移、旋转、缩放（有时仅平移/旋转）对齐，再算 MPJPE。
- **何时用**：评估姿态结构正确性，忽略绝对位置；论文中 3D 姿态估计几乎必报。

### 3. PCK（Percentage of Correct Keypoints）
- 关节误差小于阈值 \(\alpha\) 倍参照长度（如躯干长度或头部长度）即认为正确。
- **何时用**：2D/3D 检测任务，适合设置不同严格度；常与 AUC 一起使用。

### 4. AUC（Area Under the Curve）
- 以 PCK 阈值横轴、正确率为纵轴绘制曲线，求面积。
- **何时用**：需要综合评估不同阈值下表现的 ranking 任务。

### 5. Procrustes
- 对齐方法，不是最终指标。PA-MPJPE 通常调用 `scipy.spatial.procrustes` 或自定义实现。

## 参考代码片段

```python
import numpy as np
from scipy.spatial import procrustes

def mpjpe(pred, gt):
    return np.mean(np.linalg.norm(pred - gt, axis=-1))

def pa_mpjpe(pred, gt):
    _, gt_aligned, pred_aligned = procrustes(gt, pred)
    return np.mean(np.linalg.norm(gt_aligned - pred_aligned, axis=-1))
```

## 参考链接
- [Human3.6M evaluation protocol](http://vision.imar.ro/human3.6m/description.php)
- scipy Procrustes: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html
