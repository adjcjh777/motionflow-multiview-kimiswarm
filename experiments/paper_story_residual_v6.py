"""Smoke test / mini inference for the residual refinement head story.

Loads ``outputs/ray_attention_temporal_residual_v2.pth`` and runs a quick
forward pass on a synthetic clip, then optionally evaluates on a real
MPI-INF-3DHP .npz file.

Usage
-----
    conda run -n mf python experiments/paper_story_residual_v6.py
    conda run -n mf python experiments/paper_story_residual_v6.py \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_residual_model import (
    RayAttentionFusionModelTemporalResidual,
    _make_cameras,
)


def _synthetic_clip(B: int = 2, T: int = 13, V: int = 4, J: int = 17):
    """Return a random (B, T, V, J, 3) tensor and a list of Camera objects."""
    x = torch.rand(B, T, V, J, 3)
    cameras = _make_cameras(V)
    return x, cameras


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _evaluate_on_npz(model, npz_path: str, clip_len: int, device: torch.device):
    data = np.load(npz_path)
    points_2d = torch.from_numpy(data["points_2d"]).float()
    confidences = torch.from_numpy(data["confidences"]).float()
    joints_3d = torch.from_numpy(data["joints_3d"]).float()
    K = torch.from_numpy(data["camera_K"]).float()
    R = torch.from_numpy(data["camera_R"]).float()
    t = torch.from_numpy(data["camera_t"]).float()

    n_frames = points_2d.shape[0]
    # Use one centred clip for a quick read.
    start = max(0, (n_frames - clip_len) // 2)
    end = start + clip_len

    x = torch.cat(
        [points_2d[start:end], confidences[start:end].unsqueeze(-1)],
        dim=-1,
    ).unsqueeze(0)
    y = joints_3d[start:end].unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred, _ = model(
            x.to(device),
            K=K.to(device).unsqueeze(0),
            R=R.to(device).unsqueeze(0),
            t=t.to(device).unsqueeze(0),
        )
    err = (pred.cpu() - y).norm(dim=-1).mean() * 1000.0  # mm
    return err.item()


def _infer_n_views_from_checkpoint(ckpt_path: str) -> int:
    """Infer the number of views from the fusion_mlp weight shape."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    fusion_in = ckpt["fusion_mlp.0.weight"].shape[1]  # d * n_views
    d = ckpt["obs_embed.bias"].shape[0] * 2  # obs_embed output dim = d/2
    return fusion_in // d


def main():
    parser = argparse.ArgumentParser(description="Smoke test for residual refinement head")
    parser.add_argument("--val", type=str, default=None, help="Optional .npz to evaluate")
    parser.add_argument("--checkpoint", type=str, default="outputs/ray_attention_temporal_residual_v2.pth")
    parser.add_argument("--clip_len", type=int, default=13)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        n_views = _infer_n_views_from_checkpoint(str(ckpt_path))
        print(f"Inferred n_views={n_views} from checkpoint")
    else:
        n_views = 14  # MPI-INF-3DHP default
        print(f"Checkpoint not found; using n_views={n_views} for smoke test")

    # Build a residual model with the default configuration used for MPI-INF-3DHP.
    model = RayAttentionFusionModelTemporalResidual(
        j=17, d=64, n_views=n_views, n_heads=4,
        n_joint_layers=1, n_temporal_layers=2,
        max_temporal_len=256, residual_hidden=128,
    ).to(device)
    print(f"Total parameters: {_count_params(model):,}")

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"WARNING: checkpoint {ckpt_path} not found; using random weights for smoke test.")

    # 1) Synthetic smoke test.
    x, cameras = _synthetic_clip(B=2, T=args.clip_len, V=n_views, J=17)
    pred, weights = model(x.to(device), cameras=cameras)
    assert pred.shape == (2, args.clip_len, 17, 3)
    assert weights.shape == (2, args.clip_len, n_views, 17)
    print(f"Synthetic forward pass: pred={tuple(pred.shape)}, weights={tuple(weights.shape)}")

    # 2) Optional real-data quick eval.
    if args.val:
        err_mm = _evaluate_on_npz(model, args.val, args.clip_len, device)
        print(f"Quick eval MPJPE on {args.val}: {err_mm:.2f} mm")


if __name__ == "__main__":
    main()
