import sys
sys.path.insert(0, r'D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm')
import torch
from experiments.train_omniview_fusion_v5_webbridge_multi import _make_synthetic_cameras, SyntheticSmokeDataset, OmniMultiViewFusionV5

K, R, t = _make_synthetic_cameras(n_views=4)
train = SyntheticSmokeDataset(K, R, t, n_frames=50, n_joints=17, clip_len=9)
sample = train[0]
model = OmniMultiViewFusionV5(
    j=17, d=32, n_views=4, n_heads=4, n_st_layers=1, graph_num_layers=1,
    use_full_precision_dlt=True, use_robust_dlt_reweight=True,
    use_domain_embedding=True, use_camera_view_embedding=True,
    use_set_view_aggregator=True, use_entropy_regularization=True,
)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x = sample[0].unsqueeze(0).cuda()
K = sample[2].unsqueeze(0).cuda()
R = sample[3].unsqueeze(0).cuda()
t = sample[4].unsqueeze(0).cuda()
for step in range(3):
    optimizer.zero_grad()
    out = model(x, K=K, R=R, t=t, domain_id=torch.zeros(1, dtype=torch.long).cuda())
    pred, weights, visibility, L, epi = out[0], out[1], out[2], out[3], out[4]
    target = sample[1].unsqueeze(0).cuda()
    loss = torch.nn.functional.mse_loss(pred, target) + epi
    print('step', step, 'loss', loss.item(), 'pred nan', pred.isnan().any().item(), 'weights nan', weights.isnan().any().item(), 'L nan', L.isnan().any().item())
    loss.backward()
    print('  grad nan?', any(p.grad is not None and p.grad.isnan().any().item() for p in model.parameters()))
    optimizer.step()
out = model(x, K=K, R=R, t=t, domain_id=torch.zeros(1, dtype=torch.long).cuda())
print('after steps weights nan', out[1].isnan().any().item())
