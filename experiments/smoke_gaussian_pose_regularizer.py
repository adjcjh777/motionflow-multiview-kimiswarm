"""CPU smoke test: 3D Gaussian regularizer for predicted 3D pose.

Represents each predicted 3D joint as a small isotropic Gaussian, projects it
to every calibrated view, renders a low-resolution heatmap, and penalizes the
discrepancy between the rendered heatmaps and the observed 2D keypoints.

This is intentionally tiny and CPU-only: it validates the regularizer
formulation before any GPU training run is queued.

Usage
-----
    python experiments/smoke_gaussian_pose_regularizer.py
"""

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def make_circular_cameras(n_views: int = 4, radius: float = 4.0) -> tuple:
    """Return (K_list, R_list, t_list, c_list) for cameras on a circle."""
    Ks, Rs, ts, cs = [], [], [], []
    for i in range(n_views):
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0

        theta = 2.0 * math.pi * i / n_views
        phi = math.pi / 3.0
        c = radius * np.array(
            [math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)],
            dtype=np.float64,
        )
        forward = -c / (np.linalg.norm(c) + 1e-8)
        up_world = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up_world)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)  # world -> camera

        Ks.append(K)
        Rs.append(R)
        ts.append(-R @ c)
        cs.append(c)
    return Ks, Rs, ts, cs


def project_points(X: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project (J, 3) points to (J, 2) using pinhole camera."""
    Xc = (R @ X.T).T + t
    x = Xc[:, :2] / (Xc[:, 2:3] + 1e-8)
    x = (x @ K[:2, :2].T) + K[:2, 2]
    return x


# ---------------------------------------------------------------------------
# Gaussian rendering helpers
# ---------------------------------------------------------------------------

def project_gaussians(
    means: np.ndarray,
    scale: float,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project isotropic 3D Gaussians to 2D and compute per-view covariance.

    Returns
    -------
    mu2d : (J, 2)
    cov2d_inv : (J, 2, 2)
    """
    J = means.shape[0]
    Xc = (R @ means.T).T + t
    mu2d = Xc[:, :2] / (Xc[:, 2:3] + 1e-8)
    mu2d = (mu2d @ K[:2, :2].T) + K[:2, 2]

    # Simple affine approximation: covariance in image plane scales with
    # focal length and depth.  This is sufficient for a smoke test.
    depths = np.maximum(Xc[:, 2], 1e-3)
    cov2d = np.zeros((J, 2, 2), dtype=np.float64)
    for j in range(J):
        var = (scale / depths[j]) ** 2
        cov2d[j] = np.eye(2) * var
    cov2d_inv = np.linalg.inv(cov2d + 1e-4 * np.eye(2))
    return mu2d, cov2d_inv


def render_heatmap(
    mu2d: np.ndarray,
    cov2d_inv: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Render a soft heatmap by splatting 2D Gaussians over (H, W)."""
    H, W = image_size
    yy, xx = np.mgrid[0:H, 0:W]
    pixels = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float64)

    heat = np.zeros((H * W,), dtype=np.float64)
    for j in range(mu2d.shape[0]):
        diff = pixels - mu2d[j]
        exponents = -0.5 * np.sum(diff @ cov2d_inv[j] * diff, axis=1)
        exponents = np.clip(exponents, -20.0, 0.0)
        heat += np.exp(exponents)
    return heat.reshape(H, W)


def heatmap_from_keypoints(
    kp2d: np.ndarray,
    sigma: float,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Create a (H, W) heatmap by placing small Gaussians at 2D keypoints."""
    H, W = image_size
    yy, xx = np.mgrid[0:H, 0:W]
    pixels = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float64)
    heat = np.zeros((H * W,), dtype=np.float64)
    cov_inv = np.eye(2) / (sigma ** 2)
    for j in range(kp2d.shape[0]):
        diff = pixels - kp2d[j]
        exponents = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
        exponents = np.clip(exponents, -20.0, 0.0)
        heat += np.exp(exponents)
    return heat.reshape(H, W)


# ---------------------------------------------------------------------------
# Regularizer
# ---------------------------------------------------------------------------

def gaussian_pose_regularizer(
    pred_3d: np.ndarray,
    gt_3d: np.ndarray,
    cameras: list,
    image_size: tuple[int, int] = (64, 64),
    gaussian_scale: float = 0.05,
    keypoint_sigma: float = 2.0,
) -> dict:
    """Consistency loss between Gaussian-splatting render and observed keypoints.

    Args
    ----
    pred_3d : (J, 3) predicted 3D joints.
    gt_3d : (J, 3) ground-truth 3D joints (used only to derive 2D observations).
    cameras : list of (K, R, t) tuples.
    image_size : render resolution.
    gaussian_scale : world-space size of each joint Gaussian.
    keypoint_sigma : pixel std of observed keypoint heatmaps.

    Returns
    -------
    dict with total loss, per-view MSE, and opacity regularizer.
    """
    J = pred_3d.shape[0]
    losses = []
    for K, R, t in cameras:
        # Observed 2D keypoints from GT pose.
        kp2d = project_points(gt_3d, K, R, t)
        obs = heatmap_from_keypoints(kp2d, keypoint_sigma, image_size)

        # Predicted pose rendered as Gaussians.
        mu2d, cov_inv = project_gaussians(pred_3d, gaussian_scale, K, R, t, image_size)
        ren = render_heatmap(mu2d, cov_inv, image_size)

        # Normalize to unit variance so the loss is sensitive to shape mismatch.
        ren_norm = (ren - ren.mean()) / (ren.std() + 1e-8)
        obs_norm = (obs - obs.mean()) / (obs.std() + 1e-8)
        mse = float(np.mean((ren_norm - obs_norm) ** 2))
        losses.append(mse)

    return {
        "loss": np.mean(losses),
        "view_mse": np.mean(losses),
        "n_views": len(cameras),
        "n_joints": J,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(2027)
    print("3D Gaussian pose regularizer smoke test")
    print("-" * 40)

    # Synthetic humanoid skeleton (17 joints, roughly H36M scale).
    n_joints = 17
    gt_3d = rng.normal(0.0, 0.35, size=(n_joints, 3)).astype(np.float64)
    gt_3d[:, 2] += 2.5
    gt_3d[:, 1] += 0.1 * gt_3d[:, 0]  # mild correlation

    Ks, Rs, ts, _ = make_circular_cameras(n_views=4)
    cameras = list(zip(Ks, Rs, ts))

    # Case 1: perfect prediction should give near-zero loss.
    result_perfect = gaussian_pose_regularizer(
        gt_3d, gt_3d, cameras, image_size=(64, 64),
        gaussian_scale=0.05, keypoint_sigma=2.0,
    )

    # Case 2: perturbed prediction should give a larger loss.
    pred_3d = gt_3d + rng.normal(0.0, 0.05, size=gt_3d.shape)
    result_perturbed = gaussian_pose_regularizer(
        pred_3d, gt_3d, cameras, image_size=(64, 64),
        gaussian_scale=0.05, keypoint_sigma=2.0,
    )

    for name, result in [("perfect", result_perfect), ("perturbed", result_perturbed)]:
        print(f"[{name}]")
        print(f"  views:             {result['n_views']}")
        print(f"  joints:            {result['n_joints']}")
        print(f"  view_mse:          {result['view_mse']:.6f}")
        print(f"  total_loss:        {result['loss']:.6f}")

    assert result_perturbed["loss"] > result_perfect["loss"], "perturbed loss should exceed perfect loss"
    print("-" * 40)
    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
