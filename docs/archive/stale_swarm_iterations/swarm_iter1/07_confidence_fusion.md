# Confidence-weighted multiview fusion

## TL;DR
用置信度给多视角结果加权，再聚合。公式：w_i = softmax(c_i)，x_fused = Σ w_i * x_i。

## 关键结论
- 置信度作用：遮挡或检测噪声会让部分视角不可靠，加权可降低其影响。
- 2D 做法：每个视角输出 (x, y, score)，用 score 做 softmax 权重；3D 做法：三角化时用重投影误差倒数加权。
- 代表工作：Iskakov et al. ICCV 2019（可学习体素三角化）、Pavlakos et al. CVPR 2017（体素预测）、Qiu et al. ICCV 2019（跨视角特征融合）。

## 最小可运行示例（PyTorch）
~~~python
import torch, torch.nn.functional as F
xy = torch.randn(4, 17, 2)
conf = torch.rand(4, 17)
w = F.softmax(conf, dim=0).unsqueeze(-1)
fused_xy = (w * xy).sum(dim=0)
~~~

## 参考链接
- Iskakov et al.: https://arxiv.org/abs/1903.09299
- Pavlakos et al.: https://arxiv.org/abs/1611.07828
- Qiu et al.: https://arxiv.org/abs/1909.01203
