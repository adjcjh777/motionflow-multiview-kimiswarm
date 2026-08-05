"""Visualize multi-view fusion results for ray_attention_v3.

Summary of work (motionflow-multiview research swarm, turn 2026-08-04):
    Added a standalone visualization toolkit in experiments/visualize_fusion.py.
    It loads an H36M-style .npz dataset and an optional ray_attention_v3
    checkpoint, runs one frame of inference, and renders:
      1. Per-view 2D reprojections of the predicted 3D skeleton.
      2. A 3D skeleton plot comparing predicted vs ground-truth poses.
      3. A per-view per-joint attention/fusion weight heatmap.
    The script falls back to a torch-based DLT baseline when no checkpoint is
    supplied, which avoids the broken numpy/matmul SVD path observed on the
    Windows conda environment used for verification.

Renders three figure sets from a .npz multi-view dataset and an optional
ray_attention_v3 checkpoint:

1. Per-view 2D reprojections: input 2D keypoints + reprojected predicted 3D.
2. 3D skeleton: predicted vs ground-truth 3D pose.
3. Attention-weight heatmap: per-view per-joint fusion weights.

Usage:
    python experiments/visualize_fusion.py \\
        --dataset data/h36m_hf/s_01_acts_02_..._16_multiview.npz \\
        --checkpoint outputs/ray_attention_v3_h36m.pth \\
        --frame 0 \\
        --output_dir outputs/visualize_fusion

If --checkpoint is omitted, the script falls back to DLT triangulation and
confidence-scaled weights so the rendering pipeline can still be verified.

Dependencies:
    numpy, torch, matplotlib

Author: motionflow-multiview research swarm
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.ray_attention_v3_model import RayAttentionFusionModelV3

# Standard Human3.6M 17-joint skeleton used by the karfly-preprocessed subset.
# Joint order must match the prepared .npz data (pelvis is root).
JOINT_NAMES = [
    "pelvis",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "spine", "thorax", "upper_neck", "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
]

# parent[i] = index of parent joint for joint i; -1 for root.
PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ray_attention_v3 fusion outputs.")
    parser.add_argument("--dataset", type=str, default="data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview.npz",
                        help="Path to .npz with points_2d, confidences, joints_3d, camera_K/R/t.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to ray_attention_v3 .pth checkpoint. If None, use DLT baseline.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize.")
    parser.add_argument("--d", type=int, default=64, help="Model embedding dimension.")
    parser.add_argument("--output_dir", type=str, default="outputs/visualize_fusion",
                        help="Directory where figures are saved.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible camera layout.")
    return parser.parse_args()


def load_dataset(path: str):
    """Load multi-view dataset and return dict of numpy arrays plus camera list."""
    data = np.load(path)
    required = {"points_2d", "confidences", "joints_3d", "camera_K", "camera_R", "camera_t"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Dataset {path} missing keys: {missing}")

    points_2d = data["points_2d"]
    confidences = data["confidences"]
    joints_3d = data["joints_3d"]

    # Camera arrays may be (V, ...) for a single rig or (T, V, ...) per-frame.
    K = data["camera_K"]
    R = data["camera_R"]
    t = data["camera_t"]

    if K.ndim == 4:
        # Per-frame cameras; pick the requested frame.
        K = K[0]
        R = R[0]
        t = t[0]

    cameras = []
    for i in range(K.shape[0]):
        cameras.append(Camera(K=K[i], R=R[i], t=t[i]))

    return {
        "points_2d": points_2d,
        "confidences": confidences,
        "joints_3d": joints_3d,
        "cameras": cameras,
    }


def load_model(checkpoint_path: str, j: int, n_views: int, d: int, device: torch.device):
    """Load a RayAttentionFusionModelV3 checkpoint."""
    model = RayAttentionFusionModelV3(j=j, d=d, n_views=n_views).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def project_points(points_3d: np.ndarray, camera: Camera) -> np.ndarray:
    """Project (J, 3) points to 2D using a Camera object.

    Uses torch for the projection to avoid environment-specific numpy BLAS
    crashes; the result is returned as a numpy array.
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    X_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    P = torch.from_numpy(camera.projection_matrix).float()
    X_h_t = torch.from_numpy(X_h).float()
    x_h = (P @ X_h_t.T).T
    x_h = x_h / x_h[:, 2:3]
    return x_h[:, :2].cpu().numpy()


def _triangulate_point_dlt_torch(points_2d: torch.Tensor, P: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """Triangulate a single 3D point from (V, 2) and (V, 3, 4) using torch SVD.

    Args:
        points_2d: (V, 2)
        P: (V, 3, 4)
        weights: optional (V,)

    Returns:
        (3,) point.
    """
    V = points_2d.shape[0]
    if weights is None:
        weights = torch.ones(V, dtype=torch.float64)
    else:
        weights = torch.as_tensor(weights, dtype=torch.float64)
        weights = torch.sqrt(weights + 1e-6)

    A = []
    for i in range(V):
        u, v = points_2d[i]
        A.append(weights[i] * (u * P[i, 2] - P[i, 0]))
        A.append(weights[i] * (v * P[i, 2] - P[i, 1]))
    A = torch.stack(A, dim=0)

    _, _, vt = torch.linalg.svd(A)
    X = vt[-1]
    return X[:3] / X[3]


def triangulate_dlt(points_2d: np.ndarray, proj_matrices: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """Triangulate (V, J, 2) points with (V, 3, 4) projection matrices.

    Uses torch.linalg.svd internally to avoid environment-specific numpy SVD crashes.
    """
    V, J, _ = points_2d.shape
    if weights is None:
        weights = np.ones((V, J), dtype=np.float64)

    P_t = torch.from_numpy(proj_matrices).float()
    points_2d_t = torch.from_numpy(points_2d).float()
    weights_t = torch.from_numpy(weights).float()

    points_3d = np.zeros((J, 3), dtype=np.float64)
    for j_idx in range(J):
        if weights_t[:, j_idx].sum() <= 0:
            continue
        X = _triangulate_point_dlt_torch(points_2d_t[:, j_idx], P_t, weights_t[:, j_idx])
        points_3d[j_idx] = X.cpu().numpy()
    return points_3d


def _projection_matrix_torch(camera: Camera) -> torch.Tensor:
    """Return the 3x4 projection matrix as a torch tensor (no numpy matmul)."""
    K = torch.from_numpy(camera.K).float()
    R = torch.from_numpy(camera.R).float()
    t = torch.from_numpy(camera.t).float().view(3, 1)
    Rt = torch.cat([R, t], dim=1)
    return K @ Rt


def dlt_baseline(points_2d: np.ndarray, confidences: np.ndarray, cameras: list) -> np.ndarray:
    """Simple confidence-weighted DLT baseline."""
    P = torch.stack([_projection_matrix_torch(cam) for cam in cameras], dim=0).cpu().numpy()
    return triangulate_dlt(points_2d, P, weights=confidences)


def run_inference(dataset: dict, checkpoint_path: str, d: int, device: torch.device):
    """Run ray_attention_v3 on a single frame and return (pred_3d, weights)."""
    frame = dataset["frame"]
    points_2d = dataset["points_2d"][frame]
    confidences = dataset["confidences"][frame]
    cameras = dataset["cameras"]

    x = np.concatenate([points_2d, confidences[..., None]], axis=-1)
    x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(device)

    if checkpoint_path is None or not Path(checkpoint_path).exists():
        pred_3d = dlt_baseline(points_2d, confidences, cameras)
        weights = confidences.copy()
        return pred_3d, weights

    n_views = points_2d.shape[0]
    j = points_2d.shape[1]
    model = load_model(checkpoint_path, j=j, n_views=n_views, d=d, device=device)

    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().unsqueeze(0).to(device)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().unsqueeze(0).to(device)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred_3d, weights = model(x_tensor, K=K, R=R, t=t)

    pred_3d = pred_3d[0].cpu().numpy()
    weights = weights[0].cpu().numpy()
    return pred_3d, weights


def _draw_bones(ax, points_2d: np.ndarray, parents: list, color: str):
    """Draw skeleton bones on a 2D axes."""
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        xs = [points_2d[parent, 0], points_2d[child, 0]]
        ys = [points_2d[parent, 1], points_2d[child, 1]]
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.7)


def plot_multi_view_2d(dataset: dict, pred_3d: np.ndarray, output_path: Path):
    """Render per-view 2D reprojections of input keypoints and predicted 3D skeleton."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = dataset["frame"]
    points_2d = dataset["points_2d"][frame]
    confidences = dataset["confidences"][frame]
    cameras = dataset["cameras"]
    n_views = len(cameras)

    # Reproject predicted 3D skeleton into each view.
    reproj_2d = np.stack([project_points(pred_3d, cam) for cam in cameras], axis=0)

    cols = int(np.ceil(np.sqrt(n_views)))
    rows = int(np.ceil(n_views / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if n_views == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    for i, ax in enumerate(axes[:n_views]):
        ax.set_title(f"View {i}")
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")

        # Plot input 2D keypoints and skeleton.
        visible = confidences[i] > 0
        ax.scatter(points_2d[i, visible, 0], points_2d[i, visible, 1], c="blue", s=20, alpha=0.6, label="Input")
        _draw_bones(ax, points_2d[i], PARENTS, "blue")

        # Plot reprojected predicted 3D skeleton.
        ax.scatter(reproj_2d[i, :, 0], reproj_2d[i, :, 1], c="red", s=30, marker="x", alpha=0.7, label="Reproj. pred")
        _draw_bones(ax, reproj_2d[i], PARENTS, "red")

        ax.legend(loc="lower right")

    # Hide unused subplots.
    for ax in axes[n_views:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved multi-view 2D reprojection plot to {output_path}")
    plt.close(fig)


def plot_3d_skeleton(pred_3d: np.ndarray, gt_3d: np.ndarray, output_path: Path):
    """Render predicted vs ground-truth 3D skeletons."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    def _draw(ax, joints, color):
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, s=40, alpha=0.7)
        for child, parent in enumerate(PARENTS):
            if parent < 0:
                continue
            xs = [joints[parent, 0], joints[child, 0]]
            ys = [joints[parent, 1], joints[child, 1]]
            zs = [joints[parent, 2], joints[child, 2]]
            ax.plot(xs, ys, zs, color=color, linewidth=2, alpha=0.8)

    _draw(ax, gt_3d, "blue")
    _draw(ax, pred_3d, "red")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Skeleton: predicted (red) vs ground truth (blue)")
    ax.legend(["Ground truth", "Predicted"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved 3D skeleton plot to {output_path}")
    plt.close(fig)


def plot_attention_heatmap(weights: np.ndarray, output_path: Path):
    """Render per-view per-joint attention/fusion weights as a heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(weights, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(weights.shape[1]))
    ax.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(weights.shape[0]))
    ax.set_yticklabels([f"View {i}" for i in range(weights.shape[0])])
    ax.set_title("Per-view per-joint fusion weights")
    fig.colorbar(im, ax=ax, label="Weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved attention heatmap to {output_path}")
    plt.close(fig)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = load_dataset(args.dataset)
    dataset["frame"] = args.frame
    points_2d = dataset["points_2d"]
    if args.frame >= points_2d.shape[0]:
        raise ValueError(f"Frame {args.frame} out of range (0, {points_2d.shape[0] - 1})")

    pred_3d, weights = run_inference(dataset, args.checkpoint, d=args.d, device=device)
    gt_3d = dataset["joints_3d"][args.frame]

    mpjpe = np.linalg.norm(pred_3d - gt_3d, axis=-1).mean()
    print(f"Frame {args.frame}: MPJPE(pred, gt) = {mpjpe:.4f} (dataset units)")
    print(f"Per-view weight range: [{weights.min():.4f}, {weights.max():.4f}]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_multi_view_2d(dataset, pred_3d, output_dir / f"frame_{args.frame:05d}_multi_view_2d.png")
    plot_3d_skeleton(pred_3d, gt_3d, output_dir / f"frame_{args.frame:05d}_skeleton_3d.png")
    plot_attention_heatmap(weights, output_dir / f"frame_{args.frame:05d}_attention_heatmap.png")

    print(f"All figures saved to {output_dir}")


if __name__ == "__main__":
    main()
