#!/usr/bin/env python3
"""Visual diagnostic for 2/3-view catastrophic failures in variable-view inference.

Loads a trained OmniMultiViewFusion v2/v3/v4 checkpoint (or a fresh model in
smoke mode), runs variable-view inference with k=2,3,4 active views, and
produces a set of diagnostic plots together with a JSON dump of the underlying
numeric tensors.

Outputs
-------
By default writes to ``outputs/failure_analysis_variable_views/``:

* ``per_joint_error_k*.png`` – bar chart of MPJPE per joint for each k.
* ``view_weights_k*.png``      – heat-map of mean per-view triangulation weight.
* ``visibility_k*.png``        – heat-map of predicted visibility probability.
* ``triangulation_residual_k*.png`` – reprojection residual per (view, joint).
* ``summary_k*.json``          – numeric arrays backing the plots.
* ``summary.json``             – aggregated MPJPE@k and hypothesis metrics.

Usage
-----
    # CPU smoke test (synthetic data, random model)
    python scripts/visualize_variable_view_failure.py --smoke

    # Real checkpoint + dataset
    python scripts/visualize_variable_view_failure.py \
        --checkpoint outputs/omniview_fusion_v2_h36m_d128_dense_graph_a800.pth \
        --dataset data/webbridge/h36m/S9/acts_02_multiview_m.npz \
        --out_dir outputs/failure_analysis_variable_views
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import compute_all_metrics
from motionflow_mv.fusion.omniview_fusion_v2 import OmniMultiViewFusionV2
from motionflow_mv.fusion.omniview_fusion_v3 import OmniMultiViewFusionV3
from motionflow_mv.fusion.variable_view_inference import VariableViewInferenceWrapper

# Optional matplotlib import with a helpful error message.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matplotlib is required for visualize_variable_view_failure.py. "
        "Install it with: pip install matplotlib"
    ) from exc


# ---------------------------------------------------------------------------
# Synthetic smoke dataset
# ---------------------------------------------------------------------------

H36M_JOINT_NAMES = [
    "Pelvis", "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", "L_Ankle",
    "Torso", "Neck", "Nose", "Head", "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Shoulder", "R_Elbow", "R_Wrist",
]


def _make_synthetic_cameras(n_views: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        torch.from_numpy(np.stack(K_list, axis=0)).float(),
        torch.from_numpy(np.stack(R_list, axis=0)).float(),
        torch.from_numpy(np.stack(t_list, axis=0)).float(),
    )


def _project_points(
    joints_3d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Project (F, J, 3) joints through V cameras -> (F, V, J, 2)."""
    X = joints_3d
    t = t[:, None, None, :]
    X_cam = torch.einsum("vab,fjb->vfja", R, X) + t
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv = torch.matmul(K[:, None, None], (X_cam / z)[..., None])
    points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
    return points_2d.permute(1, 0, 2, 3)


class SyntheticSmokeDataset(torch.utils.data.Dataset):
    """Tiny synthetic dataset for CPU smoke test."""

    def __init__(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        n_frames: int = 60,
        n_joints: int = 17,
        clip_len: int = 9,
        noise_std: float = 0.5,
    ):
        self.K = K
        self.R = R
        self.t = t
        self.n_joints = n_joints
        self.clip_len = clip_len
        self.noise_std = noise_std

        torch.manual_seed(42)
        joints_3d = torch.randn(n_frames, n_joints, 3) * 0.3
        for _ in range(2):
            joints_3d[1:-1] = 0.5 * joints_3d[1:-1] + 0.25 * (joints_3d[:-2] + joints_3d[2:])

        points_2d = _project_points(joints_3d, K, R, t)
        if noise_std > 0:
            points_2d = points_2d + torch.randn_like(points_2d) * noise_std

        self.points_2d = points_2d
        self.confidences = torch.ones_like(points_2d[..., 0])
        self.joints_3d = joints_3d
        self.total_frames = n_frames
        self.num_clips = max(1, self.total_frames - self.clip_len + 1)

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        start = idx
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        return x, self.joints_3d[start:end], self.K, self.R, self.t


# ---------------------------------------------------------------------------
# Real dataset loader
# ---------------------------------------------------------------------------

class TemporalClipDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str, clip_len: int, stride: int = 1):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()
        self.clip_len = clip_len
        self.stride = stride
        self.total_frames = self.points_2d.shape[0]
        self.num_clips = max(1, (self.total_frames - self.clip_len) // stride + 1)

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        return x, self.joints_3d[start:end], self.K, self.R, self.t


def collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


# ---------------------------------------------------------------------------
# Model construction / loading
# ---------------------------------------------------------------------------

def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Checkpoint load: missing keys {missing[:10]}")
    if unexpected:
        print(f"Checkpoint load: unexpected keys ignored {unexpected[:10]}")


def infer_checkpoint_args(checkpoint_path: str) -> Dict[str, Any]:
    """Infer minimal architecture hyper-parameters from a saved state dict."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    view_pos_embed = state["view_pos_embed"]
    n_views, d = view_pos_embed.shape
    residual_hidden = state.get("residual_mlp.0.weight", torch.empty(128, 0)).shape[0]

    n_st_layers = 0
    while f"st_transformer.{n_st_layers}.self_attn.in_proj_weight" in state:
        n_st_layers += 1

    graph_num_layers = 1 if any(k.startswith("graph_joint_attention.") for k in state) else 0
    has_multiscale = any(k.startswith("multiscale_fusion.") for k in state)

    return {
        "d": d,
        "n_views": n_views,
        "residual_hidden": residual_hidden,
        "n_st_layers": max(1, n_st_layers),
        "graph_num_layers": graph_num_layers,
        "has_multiscale": has_multiscale,
    }


def build_model(args: argparse.Namespace, n_views: int, j: int) -> torch.nn.Module:
    use_v3 = getattr(args, "model_version", "v2") == "v3"
    cls = OmniMultiViewFusionV3 if use_v3 else OmniMultiViewFusionV2
    model = cls(
        j=j,
        d=args.d,
        n_views=n_views,
        n_heads=args.n_heads,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        graph_num_layers=args.graph_num_layers,
        n_joint_layers=args.n_joint_layers,
        return_pp_delta=False,
        return_covariance=True,
    )
    if j != 17:
        model.rebuild_graph(j, dataset="mpiinf3dhp")
    return model


# ---------------------------------------------------------------------------
# Inference and extraction
# ---------------------------------------------------------------------------

def _triangulation_reprojection_residual(
    pred_3d: torch.Tensor,
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Mean reprojection residual (V, J) averaged over batch/time for one sample.

    Args:
        pred_3d: (B, T, J, 3)
        points_2d: (B, T, V, J, 2)
        K, R, t: (V, 3, 3), (V, 3, 3), (V, 3)

    Returns:
        residuals: (V, J)
    """
    B, T, J, _ = pred_3d.shape
    V = points_2d.shape[2]
    pred_3d_flat = pred_3d.reshape(B * T, J, 3)
    points_2d_flat = points_2d.reshape(B * T, V, J, 2)

    # X_cam: (B*T, V, J, 3)
    X_cam = torch.einsum("vab,njb->nvja", R, pred_3d_flat) + t[None, :, None, :]
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv = X_cam / z  # (B*T, V, J, 3)
    uv = torch.einsum("vab,nvjb->nvja", K, uv)
    uv = uv[..., :2] / uv[..., 2:3]

    residuals = (uv - points_2d_flat).norm(dim=-1)  # (B*T, V, J)
    return residuals.mean(dim=0)  # (V, J)


def evaluate_subset(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    subset: Tuple[int, ...],
    device: torch.device,
) -> Dict[str, Any]:
    """Run inference on a fixed view subset and collect diagnostics."""
    wrapper = VariableViewInferenceWrapper(model)
    subset_list = list(subset)

    preds, gts = [], []
    per_joint_errors = []
    weights_all, visibility_all, residuals_all = [], [], []

    for xb, yb, K, R, t in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        K = K.to(device)
        R = R.to(device)
        t = t.to(device)

        with torch.no_grad():
            pred, weights, visibility, *_ = wrapper(xb, K, R, t, active_views=subset_list)

        preds.append(pred.cpu().numpy())
        gts.append(yb.cpu().numpy())

        # Per-joint error in the same units as the data (assumed metres -> mm).
        diff = (pred - yb).detach()
        per_joint_errors.append(diff.norm(dim=-1).cpu().numpy() * 1000.0)

        # Average weights/visibility over batch/time -> (V, J).
        weights_all.append(weights.mean(dim=(0, 1)).cpu().numpy())
        visibility_all.append(visibility.mean(dim=(0, 1)).cpu().numpy())

        # Triangulation reprojection residual.
        residuals = _triangulation_reprojection_residual(pred, xb[..., :2], K[0], R[0], t[0])
        residuals_all.append(residuals.cpu().numpy())

    preds_arr = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], 3) * 1000.0
    gts_arr = np.concatenate(gts, axis=0).reshape(-1, gts[0].shape[-2], 3) * 1000.0
    report = compute_all_metrics(preds_arr, gts_arr)

    per_joint_errors_arr = np.concatenate(per_joint_errors, axis=0).reshape(-1, per_joint_errors[0].shape[-1])
    per_joint_mpjpe = per_joint_errors_arr.mean(axis=0)

    return {
        "mpjpe": float(report["mpjpe"]),
        "pa_mpjpe": float(report["pa_mpjpe"]),
        "per_joint_mpjpe_mm": per_joint_mpjpe.tolist(),
        "view_weights": np.stack(weights_all, axis=0).mean(axis=0).tolist(),
        "visibility": np.stack(visibility_all, axis=0).mean(axis=0).tolist(),
        "reprojection_residual": np.stack(residuals_all, axis=0).mean(axis=0).tolist(),
        "n_frames": preds_arr.shape[0],
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_per_joint_error(per_joint_mpjpe: np.ndarray, k: int, out_path: Path) -> None:
    plt.figure(figsize=(12, 5))
    x = np.arange(len(per_joint_mpjpe))
    plt.bar(x, per_joint_mpjpe)
    plt.xticks(x, H36M_JOINT_NAMES, rotation=45, ha="right")
    plt.ylabel("MPJPE (mm)")
    plt.title(f"Per-joint MPJPE with k={k} active views")
    plt.grid(axis="y", alpha=0.3)
    _savefig(out_path)


def plot_heatmap(data: np.ndarray, k: int, title: str, out_path: Path, cmap: str = "viridis") -> None:
    plt.figure(figsize=(10, 5))
    plt.imshow(data, aspect="auto", cmap=cmap)
    plt.colorbar()
    plt.xlabel("Joint index")
    plt.ylabel("View index")
    plt.title(title)
    n_views, n_joints = data.shape
    # Annotate active vs inactive views.
    for v in range(n_views):
        alpha = 1.0 if v < k else 0.3
        plt.axhline(v - 0.5, color="white", alpha=alpha, linewidth=0.5)
    _savefig(out_path)


def plot_residual_distribution(residuals: np.ndarray, k: int, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(residuals.flatten(), bins=50, edgecolor="black")
    plt.xlabel("Reprojection residual (px)")
    plt.ylabel("Frequency")
    plt.title(f"Triangulation residual distribution (k={k})")
    plt.yscale("log")
    plt.grid(axis="y", alpha=0.3)
    _savefig(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual diagnostic for variable-view failure cases",
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to OmniMultiViewFusion v2/v3 checkpoint")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to .npz dataset (must contain points_2d, confidences, joints_3d, camera_K/R/t)")
    parser.add_argument("--model_version", type=str, default="v2", choices=["v2", "v3"],
                        help="Whether the checkpoint is a v2 or v3 model")
    parser.add_argument("--smoke", action="store_true", help="Run CPU smoke test with synthetic data")
    parser.add_argument("--clip_len", type=int, default=9, help="Temporal clip length")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--out_dir", type=str, default="outputs/failure_analysis_variable_views",
                        help="Directory to write figures and JSON summaries")
    parser.add_argument("--d", type=int, default=128, help="Model feature dimension")
    parser.add_argument("--residual_hidden", type=int, default=256, help="Residual MLP hidden size")
    parser.add_argument("--n_st_layers", type=int, default=3, help="Spatio-temporal transformer layers")
    parser.add_argument("--graph_num_layers", type=int, default=1, help="Graph-joint attention layers")
    parser.add_argument("--n_joint_layers", type=int, default=1, help="Dense joint-level layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    args = parser.parse_args()

    if args.checkpoint and Path(args.checkpoint).exists():
        inferred = infer_checkpoint_args(args.checkpoint)
        args.d = inferred["d"]
        args.residual_hidden = inferred["residual_hidden"]
        args.n_st_layers = inferred["n_st_layers"]
        args.graph_num_layers = inferred["graph_num_layers"]
        if inferred.get("has_multiscale"):
            args.model_version = "v3"

    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if args.smoke or args.dataset is None:
        print("Smoke mode: using synthetic dataset")
        n_views = 4
        n_joints = 17
        K, R, t = _make_synthetic_cameras(n_views)
        dataset = SyntheticSmokeDataset(K, R, t, n_frames=60, n_joints=n_joints, clip_len=args.clip_len)
    else:
        data = np.load(args.dataset)
        n_views = int(data["camera_K"].shape[0])
        n_joints = int(data["points_2d"].shape[2])
        dataset = TemporalClipDataset(args.dataset, args.clip_len)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(args, n_views=n_views, j=n_joints).to(device)
    if args.checkpoint:
        if not Path(args.checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        load_checkpoint(model, args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided; using freshly initialised model")
    model.eval()

    # ------------------------------------------------------------------
    # Evaluate k=2,3,4 using the canonical subset of the first k views.
    # ------------------------------------------------------------------
    summary: Dict[str, Any] = {}
    per_k_results: Dict[str, Any] = {}

    for k in [2, 3, 4]:
        if k > n_views:
            continue
        subset = tuple(range(k))
        print(f"Evaluating k={k} ...")
        result = evaluate_subset(model, loader, subset, device)
        per_k_results[str(k)] = result

        # Save per-k JSON.
        with open(out_dir / f"summary_k{k}.json", "w") as f:
            json.dump(result, f, indent=2)

        # Plots.
        per_joint = np.asarray(result["per_joint_mpjpe_mm"])
        plot_per_joint_error(per_joint, k, out_dir / f"per_joint_error_k{k}.png")

        weights = np.asarray(result["view_weights"])
        plot_heatmap(weights, k, f"Mean triangulation weights (k={k})",
                     out_dir / f"view_weights_k{k}.png", cmap="magma")

        visibility = np.asarray(result["visibility"])
        plot_heatmap(visibility, k, f"Mean predicted visibility (k={k})",
                     out_dir / f"visibility_k{k}.png", cmap="coolwarm")

        residuals = np.asarray(result["reprojection_residual"])
        plot_heatmap(residuals, k, f"Mean reprojection residual (px) (k={k})",
                     out_dir / f"triangulation_residual_k{k}.png", cmap="inferno")

        plot_residual_distribution(residuals, k, out_dir / f"residual_distribution_k{k}.png")

        summary[f"k={k}"] = {
            "mpjpe_mm": result["mpjpe"],
            "pa_mpjpe_mm": result["pa_mpjpe"],
            "max_per_joint_error_mm": float(per_joint.max()),
            "mean_active_weight": float(weights[:k].mean()),
            "mean_inactive_weight": float(weights[k:].mean()) if k < n_views else 0.0,
            "mean_active_visibility": float(visibility[:k].mean()),
            "mean_inactive_visibility": float(visibility[k:].mean()) if k < n_views else 0.0,
            "mean_residual_active_px": float(residuals[:k].mean()),
            "mean_residual_inactive_px": float(residuals[k:].mean()) if k < n_views else 0.0,
        }

        print(f"  k={k}: MPJPE={result['mpjpe']:.2f} mm, PA-MPJPE={result['pa_mpjpe']:.2f} mm, "
              f"max per-joint error={per_joint.max():.2f} mm")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFigures and JSON saved to: {out_dir}")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
