import torch
from motionflow_mv.losses import reprojection_loss
B,T,J,V = 2,3,28,14
pred = torch.randn(B,T,J,3).cuda()
points_2d = torch.randn(B,T,V,J,2).cuda()
K = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B,V,3,3).cuda()
R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B,V,3,3).cuda()
t = torch.zeros(B,V,3).cuda()
conf = torch.ones(B,T,V,J).cuda()
loss = reprojection_loss(pred, points_2d, K, R, t, confidences=conf)
print("loss", loss.item(), "shape ok")
