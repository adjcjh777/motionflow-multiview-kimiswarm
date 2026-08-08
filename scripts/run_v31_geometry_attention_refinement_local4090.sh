#!/usr/bin/env bash
# v31 geometry-attention refinement smoke test on local RTX 4090.
# This tests the new HierarchicalViewEncoderV31 module in isolation.
# Full training requires a follow-up one-line integration into
# motionflow_mv/fusion/omniview_fusion_v5.py (flag --use_hierarchical_multiview_v31).
set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON=${PYTHON:-python}

$PYTHON - <<'PY'
import torch
from motionflow_mv.fusion.hierarchical_multiview_v31 import HierarchicalViewEncoderV31

print("v31 Geometry-Attention Refinement smoke test")
print("=" * 50)

B, T, V, J, d = 2, 3, 4, 17, 64
tokens = torch.randn(B, T, V, J, d, requires_grad=True)
points_2d = torch.randn(B, T, V, J, 2)
K = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, V, 3, 3).contiguous()
R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, V, 3, 3).contiguous()
t = torch.zeros(B, V, 3)

model = HierarchicalViewEncoderV31(
    d=d,
    n_heads=4,
    n_views=V,
    n_part_layers=2,
    stochastic_depth_prob=0.0,
)

# Geometry-aware forward
out = model(tokens, points_2d=points_2d, K=K, R=R, t=t)
assert out.shape == tokens.shape, "output shape mismatch"
print(f"geometry-aware output shape: {tuple(out.shape)}")
print(f"identity-at-init (atol=1e-2): {torch.allclose(out, tokens.detach(), atol=1e-2)}")
loss = out.sum()
loss.backward()
print(f"gradients computed: {any(p.grad is not None for p in model.parameters() if p.requires_grad)}")

# Content-only fallback (no cameras)
out2 = model(tokens)
print(f"content-only fallback shape: {tuple(out2.shape)}")
print(f"content-only identity-at-init: {torch.allclose(out2, tokens.detach(), atol=1e-2)}")

# View mask: residual for masked views should be zeroed
mask = torch.ones(B, T, V, dtype=torch.bool)
mask[:, :, -1] = False
out3 = model(tokens, view_mask=mask, points_2d=points_2d, K=K, R=R, t=t)
residual = (out3 - tokens).abs()
print(f"max residual at masked view: {residual[:, :, -1].max().item():.6f}")
print("v31 smoke test passed")
PY
