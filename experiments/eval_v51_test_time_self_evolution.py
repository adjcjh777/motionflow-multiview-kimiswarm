"""Smoke / mini-evaluation for the v51 Test-Time Self-Evolution Refiner.

Loads or creates an OmniMultiViewFusionV5 model, runs inference on a small
WebBridge clip with and without ``use_v51_test_time_self_evolution_refiner``,
and reports the MPJPE plus the per-view reliability and per-joint uncertainty
exposed by the model.

Usage
-----
    # Use the v46 A800 checkpoint on an H36M validation clip
    python experiments/eval_v51_test_time_self_evolution.py \
        --checkpoint outputs/omniview_fusion_v46_sparse_view_generalization_on_v45_a800.pth \
        --npz data/webbridge/h36m_meters/s_09_acts_02_multiview_m.npz \
        --clip_len 9

    # Fresh synthetic smoke (no checkpoint required)
    python experiments/eval_v51_test_time_self_evolution.py --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.omniview_fusion_v5 import OmniMultiViewFusionV5


def _make_synthetic_cameras(n_views: int = 4):
    K_list, R_list, t_list = [], [], []
    for i in range(n_views):
        theta = 2 * math.pi * i / n_views
        c = np.array([3.0 * np.cos(theta), 3.0 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        K_list.append(K)
        R_list.append(R)
        t_list.append(t)
    return (
        torch.from_numpy(np.stack(K_list)).float(),
        torch.from_numpy(np.stack(R_list)).float(),
        torch.from_numpy(np.stack(t_list)).float(),
    )


def _mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return torch.norm(pred - gt, dim=-1).mean().item()


def run_inference(
    model: OmniMultiViewFusionV5,
    x: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model(x, K=K, R=R, t=t)
    return out[0]


def evaluate_clip(
    model: OmniMultiViewFusionV5,
    x: torch.Tensor,
    y: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    tag: str,
) -> dict:
    pred = run_inference(model, x, K, R, t)
    err = _mpjpe(pred, y)
    result = {"tag": tag, "mpjpe_mm": err * 1000.0, "pred_shape": list(pred.shape)}
    if hasattr(model, "v51_tta_last_reliability") and model.v51_tta_last_reliability is not None:
        rel = model.v51_tta_last_reliability
        result["reliability_mean"] = rel.mean().item()
        result["reliability_std"] = rel.std().item()
    if hasattr(model, "v51_tta_last_uncertainty") and model.v51_tta_last_uncertainty is not None:
        unc = model.v51_tta_last_uncertainty
        result["uncertainty_mean"] = unc.mean().item()
        result["uncertainty_std"] = unc.std().item()
    return result


def build_smoke_model() -> OmniMultiViewFusionV5:
    return OmniMultiViewFusionV5(
        j=17,
        n_views=4,
        d=32,
        n_st_layers=1,
        residual_hidden=64,
        principal_point_hidden=32,
        use_v51_test_time_self_evolution_refiner=True,
        v51_tta_num_steps=3,
    )


def build_smoke_batch(clip_len: int = 9):
    B = 1
    V = 4
    J = 17
    K, R, t = _make_synthetic_cameras(V)
    # Random 3-D pose in metres
    y = torch.randn(B, clip_len, J, 3)
    # Project to 2-D
    X_cam = torch.einsum("vil,btjl->btjvi", R, y) + t[None, None, None, :]
    z = X_cam[..., 2:3]
    xy = X_cam[..., :2] / (z + 1e-6)
    K_block = K[None, None, None, :, :2, :2]
    proj = (K_block @ xy.unsqueeze(-1)).squeeze(-1) + K[None, None, None, :, :2, 2]
    proj = proj.permute(0, 1, 3, 2, 4)
    conf = torch.ones(B, clip_len, V, J, 1)
    x = torch.cat([proj, conf], dim=-1)
    return x, y, K[None].expand(B, -1, -1, -1), R[None].expand(B, -1, -1, -1), t[None].expand(B, -1, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="v51 TTSER mini-evaluation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--npz", type=str, default=None)
    parser.add_argument("--clip_len", type=int, default=9)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tta_num_steps", type=int, default=3)
    parser.add_argument("--tta_num_steps_baseline", type=int, default=0, help="0 disables TTSER")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.smoke:
        model = build_smoke_model()
        x, y, K, R, t = build_smoke_batch(args.clip_len)
    else:
        raise NotImplementedError("Real-data eval not yet implemented; use --smoke")

    model = model.to(args.device)
    x = x.to(args.device)
    y = y.to(args.device)
    K = K.to(args.device)
    R = R.to(args.device)
    t = t.to(args.device)

    results = []

    # Baseline without TTSER
    model.use_v51_test_time_self_evolution_refiner = False
    results.append(evaluate_clip(model, x, y, K, R, t, tag="baseline_no_tta"))

    # With TTSER
    model.use_v51_test_time_self_evolution_refiner = True
    results.append(evaluate_clip(model, x, y, K, R, t, tag="v51_tta"))

    for r in results:
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
