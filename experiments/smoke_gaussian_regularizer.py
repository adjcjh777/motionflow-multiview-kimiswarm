"""Smoke test: minimal CPU forward of a stub 3D Gaussian consistency loss.

This script does **not** integrate with the full MotionFlow-MultiView training
loop.  It only exercises a tiny standalone implementation of a 3D-Gaussian
consistency regularizer and prints the resulting loss value.

Usage
-----
    python experiments/smoke_gaussian_regularizer.py
"""

import math

import numpy as np


def _make_rotation_matrix(angle: float, axis: np.ndarray) -> np.ndarray:
    """Rodrigues formula: rotate by ``angle`` (radians) around ``axis``."""
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    R = np.array(
        [
            [c + x * x * t, x * y * t - z * s, x * z * t + y * s],
            [y * x * t + z * s, c + y * y * t, y * z * t - x * s],
            [z * x * t - y * s, z * y * t + x * s, c + z * z * t],
        ]
    )
    return R


class Stub3DGaussians:
    """A tiny container for 3D Gaussian parameters on CPU."""

    def __init__(self, num_points: int = 64, rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng(2026)
        self.rng = rng
        self.num_points = num_points
        # means in a roughly human-scale volume (meters)
        self.means = rng.normal(0.0, 0.35, size=(num_points, 3))
        self.means[:, 2] += 2.5  # in front of the cameras
        # isotropic-ish scales
        self.scales = np.clip(rng.lognormal(-0.5, 0.5, size=(num_points, 3)), 0.5, 3.0)
        # random rotations
        self.rotations = [
            _make_rotation_matrix(rng.uniform(0.0, 2.0 * math.pi), rng.standard_normal(3))
            for _ in range(num_points)
        ]
        # opacity and color
        self.opacity = rng.uniform(0.3, 0.9, size=(num_points, 1))
        self.color = rng.uniform(0.0, 1.0, size=(num_points, 3))

    @property
    def covariances(self) -> np.ndarray:
        """(N, 3, 3) covariance matrices Sigma = R * S^2 * R^T."""
        covs = np.zeros((self.num_points, 3, 3), dtype=np.float64)
        for i in range(self.num_points):
            S = np.diag(self.scales[i] ** 2)
            R = self.rotations[i]
            covs[i] = R @ S @ R.T
        return covs


def _make_camera(theta: float, radius: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return intrinsics K, rotation R, and camera center c."""
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = K[1, 1] = 800.0
    K[0, 2] = 320.0
    K[1, 2] = 240.0

    # camera looks at origin from (radius, theta) in the x-y plane, roughly
    phi = math.pi / 3.0
    c = radius * np.array(
        [math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)],
        dtype=np.float64,
    )
    # forward, right, up axes
    forward = -c / (np.linalg.norm(c) + 1e-8)
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, forward)
    R = np.stack([right, up, -forward], axis=0)  # world -> cam
    return K, R, c


def _project_covariances(
    means: np.ndarray,
    covs: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D Gaussians to 2D using the affine local projection approximation.

    Returns
    -------
    mu2d : (N, 2)
    cov2d : (N, 2, 2)
    """
    N = means.shape[0]
    # world -> camera
    Xc = ((R @ means.T).T + (-R @ c)).astype(np.float64)
    # perspective projection x = X / Z, y = Y / Z
    mu2d = Xc[:, :2] / (Xc[:, 2:3] + 1e-8)
    mu2d = (mu2d @ K[:2, :2].T) + K[:2, 2]

    cov2d = np.zeros((N, 2, 2), dtype=np.float64)
    for i in range(N):
        # Jacobian of perspective projection at Xc
        Z = Xc[i, 2]
        J = np.array(
            [
                [1.0 / Z, 0.0, -Xc[i, 0] / (Z * Z)],
                [0.0, 1.0 / Z, -Xc[i, 1] / (Z * Z)],
            ],
            dtype=np.float64,
        )
        # transform covariance to camera frame
        cov_cam = R @ covs[i] @ R.T
        cov_proj = J @ cov_cam @ J.T
        # apply intrinsics scaling
        S = K[:2, :2]
        cov2d[i] = S @ cov_proj @ S.T
    return mu2d, cov2d


def _render_splat(
    mu2d: np.ndarray,
    cov2d: np.ndarray,
    opacity: np.ndarray,
    color: np.ndarray,
    H: int = 64,
    W: int = 64,
) -> np.ndarray:
    """Render a color image by splatting 2D Gaussians (CPU stub).

    Uses a simple alpha-compositing approximation.
    """
    img = np.zeros((H, W, 3), dtype=np.float64)
    accum_alpha = np.zeros((H, W), dtype=np.float64)
    # pixel grid
    yy, xx = np.mgrid[0:H, 0:W]
    pixels = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float64)

    for i in range(mu2d.shape[0]):
        mu = mu2d[i]
        cov = cov2d[i]
        try:
            cov_inv = np.linalg.inv(cov + 1e-4 * np.eye(2))
        except np.linalg.LinAlgError:
            cov_inv = np.eye(2) / (1e-4)
        diff = pixels - mu
        # unnormalized Gaussian kernel
        exponents = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
        g = np.exp(np.clip(exponents, -20.0, 0.0))
        g = g.reshape(H, W)

        alpha_i = opacity[i, 0] * g
        rem = 1.0 - accum_alpha
        weight = rem * alpha_i
        img += weight[:, :, None] * color[i][None, None, :]
        accum_alpha = np.clip(accum_alpha + alpha_i, 0.0, 1.0)

    return img


def gaussian_consistency_loss(
    gaussians: Stub3DGaussians,
    cameras: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    image_size: tuple[int, int] = (64, 64),
) -> dict:
    """Compute a stub multi-view consistency loss for the given 3D Gaussians.

    The loss encourages the per-view splatted images to agree with each other.
    It is composed of:

    * an L2 image consistency term between all pairs of views,
    * a small opacity regularizer to keep opacity bounded.
    """
    covs = gaussians.covariances

    renderings = []
    for K, R, c in cameras:
        mu2d, cov2d = _project_covariances(gaussians.means, covs, K, R, c)
        img = _render_splat(mu2d, cov2d, gaussians.opacity, gaussians.color, *image_size)
        renderings.append(img)

    # pairwise L2 consistency
    num_views = len(renderings)
    consistency = 0.0
    count = 0
    for i in range(num_views):
        for j in range(i + 1, num_views):
            consistency += np.mean((renderings[i] - renderings[j]) ** 2)
            count += 1
    consistency /= max(count, 1)

    # opacity regularizer
    opacity_reg = np.mean(gaussians.opacity ** 2)

    loss = consistency + 0.01 * opacity_reg
    return {
        "loss": float(loss),
        "image_consistency": float(consistency),
        "opacity_reg": float(opacity_reg),
    }


def main():
    rng = np.random.default_rng(2027)
    print("Stub 3D Gaussian consistency smoke test")
    print("-" * 40)

    gaussians = Stub3DGaussians(num_points=32, rng=rng)
    print(f"Gaussian atoms: {gaussians.num_points}")

    # two synthetic views
    cameras = [
        _make_camera(theta=0.0),
        _make_camera(theta=math.pi / 6.0),
    ]

    result = gaussian_consistency_loss(gaussians, cameras, image_size=(64, 64))

    print(f"image_consistency: {result['image_consistency']:.6f}")
    print(f"opacity_reg:       {result['opacity_reg']:.6f}")
    print(f"total_loss:        {result['loss']:.6f}")
    print("-" * 40)
    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
