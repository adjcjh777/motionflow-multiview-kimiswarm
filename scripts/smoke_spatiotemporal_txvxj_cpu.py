"""CPU-only smoke test for the (T x V x J) spatio-temporal direction.

Compares the existing factorised ``SpatiotemporalPrincipalPointModel`` against the
current best PP baseline ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint``
on synthetic data.  No training and no GPU are used, so it is safe to run while the
RTX 4090 is busy with the cross-view PP curriculum.
"""

import os
import sys
import time

# Safe on the Anaconda/MKL stack used in this repo.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from motionflow_mv.models.spatiotemporal_principal_point_model import (
    SpatiotemporalPrincipalPointModel,
    _make_cameras,
)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def time_forward_backward(model, x, cameras, n_iter=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        pred = model(x, cameras=cameras)[0]
        loss = pred.mean()
        loss.backward()
        optimizer.step()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times), times


def main():
    device = torch.device("cpu")
    B, T, V, J = 2, 13, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    print("=== (T x V x J) spatio-temporal smoke test (CPU) ===")
    print(f"batch={B}, clip_len={T}, views={V}, joints={J}")
    print()

    baseline = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=J,
        d=64,
        n_views=V,
        n_heads=4,
        n_joint_layers=1,
        n_st_layers=2,
        residual_hidden=128,
        return_pp_delta=True,
    ).to(device)

    st_model = SpatiotemporalPrincipalPointModel(
        j=J,
        d=64,
        n_views=V,
        n_heads=4,
        n_temporal_layers=1,
        n_view_layers=1,
        n_joint_layers=1,
        residual_hidden=128,
        return_pp_delta=True,
    ).to(device)

    print(f"Baseline params: {parameter_count(baseline):,}")
    print(f"Factorised T x V x J params: {parameter_count(st_model):,}")
    print()

    with torch.no_grad():
        pred_b = baseline(x, cameras=cameras)[0]
        pred_s = st_model(x, cameras=cameras)[0]
    assert pred_b.shape == (B, T, J, 3)
    assert pred_s.shape == (B, T, J, 3)
    print("Forward shapes OK.")

    baseline_avg, _ = time_forward_backward(baseline, x, cameras)
    st_avg, _ = time_forward_backward(st_model, x, cameras)

    print(f"Baseline forward+backward: {baseline_avg:.3f}s/iter")
    print(f"Factorised T x V x J forward+backward: {st_avg:.3f}s/iter")
    print()

    ratio = st_avg / baseline_avg
    print(f"Relative CPU time (T x V x J / baseline): {ratio:.2f}x")

    if ratio > 3.0:
        print("WARNING: factorised model is materially slower on CPU; profile GPU.")
    else:
        print("Factorised model runs within ~3x of the baseline on CPU.")


if __name__ == "__main__":
    main()
