# Lightweight attention fusion for multiview skeletons

## TL;DR
在多视角骨架任务中，把每个关节的各视角特征看做一个序列，用轻量 attention 学习视角权重并加权聚合。它能自动抑制遮挡/检测差的视角，计算量仅随视角数线性增长，适合替换当前 mean/confidence 融合层。

## 关键结论
1. 核心思想：对每个关节 j，用可学习的 query 与各视角 key 做点积，softmax 得到视角权重，再对 value 加权求和。
2. 轻量做法：单头、低维 d=32~64，每个关节独立计算；无需完整 Transformer，参数量 O(d^2)，计算量 O(V J d^2)。
3. 相关论文：
   - Shuai et al., Adaptive Multi-view and Temporal Fusing Transformer for 3D Human Pose Estimation, arXiv:2110.05092. 提出 MFT，用相对注意力处理可变视角数。
   - Huang et al., VTP: Volumetric Transformer for Multi-view Multi-person 3D Pose Estimation, arXiv:2205.12602. 用 3D 体素 Transformer，并引入稀疏 Sinkhorn attention 降低复杂度。
   - Moliner et al., Geometry-Biased Transformer for Robust Multi-View 3D Human Pose Reconstruction, arXiv:2312.17106. 几何偏置 attention，对遮挡/少视角更鲁棒。
   - Hu et al., ACRNet: Attention Cube Regression Network for Multi-view Real-time 3D Human Pose Estimation, arXiv:2210.05130. 在 cube 表面用 attention 点回归关节。
4. 复杂度：设 V 为视角数、J 为关节数、d 为特征维度。
   - 参数量：3 d^2 + J d（query 嵌入）。
   - 单帧计算：O(V J d^2)。
   - 当 V <= 8、d <= 64 时，额外开销远小于 backbone。

## 架构草图
~~~
Input: x[v, j] in R^d  (v = 1..V views, j = 1..J joints)
For each joint j:
  q_j   = W_q * p_j          # p_j: learnable joint query
  k_v,j = W_k * x[v, j]
  v_v,j = W_v * x[v, j]
  a_v,j = softmax_v( q_j^T k_v,j / sqrt(d) )
  y_j   = sum_v a_v,j * v_v,j
Output: y_j in R^d
~~~

## 最小可运行示例（PyTorch）
~~~python
import torch
import torch.nn as nn
import math

class ViewAttentionFusion(nn.Module):
    def __init__(self, d=32, j=17):
        super().__init__()
        self.d = d
        self.query = nn.Parameter(torch.randn(j, d) * 0.02)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)

    def forward(self, x):
        # x: (B, V, J, D)
        B, V, J, D = x.shape
        q = self.query.unsqueeze(0)          # (1, J, D)
        k = self.Wk(x)                       # (B, V, J, D)
        v = self.Wv(x)                       # (B, V, J, D)
        scores = torch.einsum('b v j d, q j d -> b j v', k, q) / math.sqrt(D)
        attn = torch.softmax(scores, dim=-1) # (B, J, V)
        out = torch.einsum('b v j d, b j v -> b j d', v, attn)
        return out

# demo
x = torch.randn(2, 4, 17, 32)
y = ViewAttentionFusion(d=32, j=17)(x)
print(y.shape)   # (2, 17, 32)
~~~

## 参考链接
- arXiv:2110.05092 — Adaptive Multi-view and Temporal Fusing Transformer: https://arxiv.org/abs/2110.05092
- arXiv:2205.12602 — VTP: https://arxiv.org/abs/2205.12602
- arXiv:2312.17106 — Geometry-Biased Transformer: https://arxiv.org/abs/2312.17106
- arXiv:2210.05130 — ACRNet: https://arxiv.org/abs/2210.05130
- PyTorch MultiheadAttention: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

决策建议：在 MotionFlow 流水线中，把 per-view 特征送入上述 ViewAttentionFusion 替换 mean 或 confidence 加权，即可在不显著增加计算量的情况下提升遮挡鲁棒性。
