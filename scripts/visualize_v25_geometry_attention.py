#!/usr/bin/env python3
"""Visualise the geometry-aware cross-view attention in v25.

Loads (or synthesises) a multi-view sample, runs
``motionflow_mv.fusion.multiview_geometry_fusion_v25.MultiViewGeometryFusionV25``
with attention recording enabled, and writes:

* ``v25_attention_layer*.png`` – per-layer average attention maps for a few
  representative joints.
* ``v25_geometry_bias.png`` – the epipolar-distance and ray-intersection logits
  that bias the attention scores.
* ``v25_attention_data.npz`` – raw attention / geometry tensors for offline
  analysis.

Usage
-----
    # CPU smoke test with a freshly initialised v25 model
    python scripts/visualize_v25_geometry_attention.py

    # Visualise a trained checkpoint on a real WebBridge sample
    python scripts/visualize_v25_geometry_attention.py \
        --checkpoint outputs/v25_kap_no_ba/best.pth \
        --sample data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz \
        --frame 0
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.epipolar_attention_bias import compute_epipolar_distance
from motionflow_mv.fusion.multiview_geometry_fusion_v25 import (
    MultiViewGeometryFusionV25,
    compute_rays,
    ray_intersection_logit,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matplotlib is required for visualize_v25_geometry_attention.py. "
        "Install it with: pip install matplotlib"
    ) from exc


# Human3.6M 17-joint layout used by the synthetic/WebBridge samples.
JOINT_NAMES = [
    "pelvis", "right_hip", "right_knee", "right_ankle", "left_hip",
    "left_knee", "left_ankle", "spine", "neck", "head", "left_shoulder",
    "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
    "head_top",
]


def _make_cameras(n_views: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a simple circular camera rig (same convention as the unit tests)."""
    Ks, Rs, ts = [], [], []
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
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        Ks.append(K)
        Rs.append(R)
        ts.append(t)
    return (
        torch.from_numpy(np.stack(Ks)).float(),
        torch.from_numpy(np.stack(Rs)).float(),
        torch.from_numpy(np.stack(ts)).float(),
    )


def _project_points(
    joints_3d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Project (F, J, 3) joints through V cameras -> (F, V, J, 2)."""
    t = t[:, None, None, :]
    X_cam = torch.einsum("vab,fjb->vfja", R, joints_3d) + t
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv = torch.matmul(K[:, None, None], (X_cam / z)[..., None])
    points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
    return points_2d.permute(1, 0, 2, 3)


def make_synthetic_batch(
    B: int = 1,
    T: int = 1,
    V: int = 4,
    J: int = 17,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a synthetic (points_2d, K, R, t, view_mask) batch."""
    K, R, t = _make_cameras(V)
    torch.manual_seed(42)
    joints_3d = torch.randn(T, J, 3) * 0.3
    points_2d = _project_points(joints_3d, K, R, t)

    K = K.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
    R = R.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
    t = t.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
    points_2d = points_2d.unsqueeze(0).expand(B, -1, -1, -1, -1)
    confidence = torch.ones(B, T, V, J)
    points_2d = torch.cat([points_2d, confidence[..., None]], dim=-1)
    view_mask = torch.ones(B, T, V).bool()
    return points_2d, K, R, t, view_mask


def load_real_sample(path: str, frame: int) -> tuple[torch.Tensor, ...]:
    """Load a canonical .npz sample and return (points_2d, K, R, t, view_mask)."""
    data = np.load(path, allow_pickle=True)
    required = {"points_2d", "camera_K", "camera_R", "camera_t"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Sample {path} missing keys: {missing}")

    points_2d = torch.from_numpy(data["points_2d"]).float()
    K = torch.from_numpy(data["camera_K"]).float()
    R = torch.from_numpy(data["camera_R"]).float()
    t = torch.from_numpy(data["camera_t"]).float()

    if frame >= points_2d.shape[0]:
        raise ValueError(f"Frame {frame} out of range (0, {points_2d.shape[0] - 1})")

    points_2d = points_2d[frame : frame + 1]
    T, V, J = points_2d.shape[:3]

    # Add a confidence channel if absent.
    if points_2d.shape[-1] == 2:
        confidence = torch.ones(T, V, J, 1)
        points_2d = torch.cat([points_2d, confidence], dim=-1)

    K = K.unsqueeze(0).expand(1, -1, -1, -1)
    R = R.unsqueeze(0).expand(1, -1, -1, -1)
    t = t.unsqueeze(0).expand(1, -1, -1)
    view_mask = torch.ones(1, T, V).bool()
    return points_2d, K, R, t, view_mask


def load_checkpoint(module: torch.nn.Module, checkpoint_path: str) -> None:
    """Load a checkpoint into the v25 module, tolerating missing/extra keys."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing:
        print(f"Checkpoint load: missing keys {missing[:10]}")
    if unexpected:
        print(f"Checkpoint load: unexpected keys ignored {unexpected[:10]}")


def plot_attention_maps(
    attention: np.ndarray,
    joint_indices: list[int],
    layer_idx: int,
    out_path: Path,
) -> None:
    """Plot per-joint attention heatmaps for one layer.

    Args:
        attention: (J, V, V) average attention over heads/batch/time.
        joint_indices: which joints to visualise.
        layer_idx: 0-based layer index (used in the title).
        out_path: destination PNG path.
    """
    n_joints = len(joint_indices)
    fig, axes = plt.subplots(1, n_joints, figsize=(4 * n_joints, 4))
    if n_joints == 1:
        axes = [axes]
    V = attention.shape[1]
    for ax, j in zip(axes, joint_indices):
        im = ax.imshow(attention[j], cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"joint_{j}")
        ax.set_xlabel("Key view")
        ax.set_ylabel("Query view")
        ax.set_xticks(range(V))
        ax.set_yticks(range(V))
        for v_k in range(V):
            for v_q in range(V):
                ax.text(
                    v_k,
                    v_q,
                    f"{attention[j, v_q, v_k]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if attention[j, v_q, v_k] < 0.5 else "black",
                    fontsize=8,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"v25 geometry-aware attention – layer {layer_idx + 1}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_geometry_bias(
    epipolar_dist: np.ndarray,
    ray_logit: np.ndarray,
    joint_indices: list[int],
    out_path: Path,
) -> None:
    """Plot epipolar distance and ray-intersection logit for selected joints."""
    n_joints = len(joint_indices)
    fig, axes = plt.subplots(n_joints, 2, figsize=(8, 4 * n_joints))
    if n_joints == 1:
        axes = axes[None, :]
    for row, j in enumerate(joint_indices):
        ax0, ax1 = axes[row]
        im0 = ax0.imshow(epipolar_dist[j], cmap="magma_r")
        ax0.set_title(f"{JOINT_NAMES[j]} – epipolar distance")
        ax0.set_xlabel("Key view")
        ax0.set_ylabel("Query view")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        im1 = ax1.imshow(ray_logit[j], cmap="coolwarm")
        ax1.set_title(f"{JOINT_NAMES[j]} – ray-intersection logit")
        ax1.set_xlabel("Key view")
        ax1.set_ylabel("Query view")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise v25 geometry-aware cross-view attention."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional v25 checkpoint (.pth) to load before visualisation.",
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="Optional .npz sample to load (must contain points_2d, camera_K/R/t).",
    )
    parser.add_argument("--frame", type=int, default=0, help="Temporal frame index.")
    parser.add_argument(
        "--joints",
        type=int,
        nargs="+",
        default=None,
        help="Joint indices to plot (default: a few representative ones).",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/visualizations/v25_geometry_attention",
        help="Directory for output images and data.",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=128,
        help="Feature dimension for the v25 module.",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4,
        help="Number of attention heads in the v25 module.",
    )
    parser.add_argument(
        "--n_geometry_layers",
        type=int,
        default=2,
        help="Number of geometry-attention layers.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if args.sample:
        points_2d, K, R, t, view_mask = load_real_sample(args.sample, args.frame)
    else:
        points_2d, K, R, t, view_mask = make_synthetic_batch()

    B, T, V, J = points_2d.shape[:4]
    print(f"Visualising geometry attention: B={B}, T={T}, V={V}, J={J}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    module = MultiViewGeometryFusionV25(
        d=args.d,
        n_heads=args.n_heads,
        n_views=V,
        n_geometry_layers=args.n_geometry_layers,
        use_geometry_attention=True,
        use_learned_depth_triangulation=True,
    )
    if args.checkpoint:
        load_checkpoint(module, args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided; using freshly initialised v25 model")

    # Enable attention recording on every geometry-attention layer.
    for layer in module.geom_attn_layers:
        layer.record_attention = True
        layer.last_attention = None

    module.eval()
    with torch.no_grad():
        pred_3d, geom_loss = module(points_2d, K, R, t, view_mask=view_mask)

    print(f"Geometry reprojection loss: {geom_loss.item():.6f}")

    # ------------------------------------------------------------------
    # Extract and average attention maps.
    # Shape stored by the hook is (B*T*J, h, V, V).
    # ------------------------------------------------------------------
    attention_arrays = []
    for layer in module.geom_attn_layers:
        if layer.last_attention is None:
            raise RuntimeError("Attention was not recorded." "Did the model use geometry attention?")
        attn = layer.last_attention  # (B*T*J, h, V, V)
        # Average over heads, then reshape by batch/time/joint.
        attn = attn.mean(dim=1)  # (B*T*J, V, V)
        attn = attn.view(B, T, J, V, V)
        # Keep the first (batch, time) slice for visualisation.
        attn = attn[0, 0].numpy()  # (J, V, V)
        attention_arrays.append(attn)

    # ------------------------------------------------------------------
    # Compute the geometry biases for the same slice.
    # ------------------------------------------------------------------
    pts = points_2d[..., :2]
    epipolar_dist = compute_epipolar_distance(
        K.reshape(B * T, V, 3, 3),
        R.reshape(B * T, V, 3, 3),
        t.reshape(B * T, V, 3),
        pts.reshape(B * T, V, J, 2),
    )  # (B*T, V, V, J)
    epipolar_dist = epipolar_dist[0].permute(2, 0, 1).numpy()  # (J, V, V)

    centre, direction = compute_rays(pts, K, R, t)
    sigma_d = module.geom_attn_layers[0].sigma_d
    sigma_a = module.geom_attn_layers[0].sigma_a
    ray_logit = ray_intersection_logit(centre, direction, sigma_d, sigma_a)
    ray_logit = ray_logit[0, 0].detach().permute(2, 0, 1).numpy()  # (J, V, V)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    if args.joints is None:
        # Default to a few representative joints covering torso + limbs.
        default_joints = [0, 2, 8, 11, 14]
        joint_indices = [j for j in default_joints if j < J]
    else:
        joint_indices = [j for j in args.joints if 0 <= j < J]

    if not joint_indices:
        raise ValueError(f"No valid joint indices provided for J={J}")

    for layer_idx, attn in enumerate(attention_arrays):
        plot_attention_maps(
            attn,
            joint_indices,
            layer_idx,
            out_dir / f"v25_attention_layer{layer_idx + 1}.png",
        )

    plot_geometry_bias(
        epipolar_dist,
        ray_logit,
        joint_indices,
        out_dir / "v25_geometry_bias.png",
    )

    np.savez(
        out_dir / "v25_attention_data.npz",
        attention=np.stack(attention_arrays, axis=0),  # (L, J, V, V)
        epipolar_dist=epipolar_dist,
        ray_logit=ray_logit,
        joint_indices=np.array(joint_indices),
    )

    print(f"Saved attention and geometry-bias visualisations to {out_dir}")


if __name__ == "__main__":
    main()
