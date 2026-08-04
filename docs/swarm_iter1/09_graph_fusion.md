# 基于图神经网络的多视角姿态融合调研

## TL;DR
将多视角人体姿态估计建模为图：节点同时包含 camera view 与人体关节，边表示“同一视角内骨骼连接”或“不同视角同一关节”。通过 GNN 消息传递聚合跨视角、跨关节信息，可提升遮挡与视角差异下的姿态估计精度。

## 关键结论

1. **图建模方式**
   - **异构/二部图**：view 节点负责聚合相机视角信息，joint 节点负责骨骼结构约束。
   - **纯关节图**：每个视角下的 2D 关节作为节点，跨视角同关节连边 + 同视角骨骼连边，直接做消息传递。

2. **消息传递目标**
   - **跨视角一致性**：让不同 camera 对同一关节的观测互相校正。
   - **跨关节一致性**：利用人体骨骼拓扑抑制孤立噪声与异常检测。

3. **最小可运行示例（纯 PyTorch）**

       import torch
       import torch.nn as nn
       import torch.nn.functional as F

       class ViewJointGNN(nn.Module):
           def __init__(self, in_dim, hidden_dim):
               super().__init__()
               self.edge_mlp = nn.Sequential(
                   nn.Linear(in_dim * 2, hidden_dim), nn.ReLU(),
                   nn.Linear(hidden_dim, 1)
               )
               self.update = nn.Sequential(
                   nn.Linear(in_dim + hidden_dim, hidden_dim), nn.ReLU()
               )

           def forward(self, x, edges):
               # x: (N, in_dim) 节点特征
               # edges: (E, 2) 边索引
               src, dst = edges[:, 0], edges[:, 1]
               a = torch.sigmoid(self.edge_mlp(torch.cat([x[src], x[dst]], -1)))
               msg = a * x[src]
               aggr = torch.zeros_like(x).index_add_(0, dst, msg)
               return self.update(torch.cat([x, aggr], -1))

4. **落地建议**
   - 先用纯关节图 + 跨视角同关节连边做 baseline，复用现有 2D detector。
   - 视角/关节数量大时，再引入 view 节点做异构图，避免全连接导致计算爆炸。
   - 可与三角化（triangulation）互补：GNN 输出置信度可作为加权 triangulation 的权重。

## 参考链接/代码片段

- Microsoft Graph-based Multi-view 3D Human Pose Estimation：https://github.com/microsoft/Graph-based-Multi-view-3D-Human-Pose-Estimation
- PyTorch Geometric Message Passing 教程：https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html
- ST-GCN（骨骼图卷积）：https://github.com/yysijie/ST-GCN
