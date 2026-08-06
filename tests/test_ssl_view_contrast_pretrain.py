"""CPU smoke test for self-supervised pretraining with cross-view contrastive loss.

Generates a tiny synthetic multi-view ``.npz`` dataset and runs a single epoch of
``experiments/pretrain_ray_attention_ssl.py`` with ``--lambda_contrast 0.1`` on the
CPU.  The test should finish in well under two minutes.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_SCRIPT = ROOT / "experiments" / "pretrain_ray_attention_ssl.py"


def _make_cameras(n_views: int = 4, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng(2024)
    cameras = []
    for i in range(n_views):
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 800.0
        K[:2, 2] = rng.uniform(300, 340, size=2)
        R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        theta = 2 * np.pi * i / n_views
        phi = np.pi / 3
        radius = 5.0
        c = radius * np.array(
            [
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ]
        )
        t = -R @ c
        cameras.append((K, R, t))
    return cameras


def _project(X: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project (J, 3) world points to (J, 2) using K, R, t."""
    X_cam = (R @ X.T).T + t
    x = X_cam[:, :2] / (X_cam[:, 2:3] + 1e-6)
    Kx = (K @ np.hstack([x, np.ones((X.shape[0], 1))]).T).T
    return Kx[:, :2]


def _random_skeleton(j: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.normal(0.0, 0.15, size=(j, 3))
    X[:, 2] += 2.0
    return X


def _write_synthetic_npz(path: Path, n_frames: int, n_views: int, j: int):
    rng = np.random.default_rng(42)
    cameras = _make_cameras(n_views, rng)
    Ks = np.stack([c[0] for c in cameras], axis=0).astype(np.float32)
    Rs = np.stack([c[1] for c in cameras], axis=0).astype(np.float32)
    ts = np.stack([c[2] for c in cameras], axis=0).astype(np.float32)

    points_2d = np.zeros((n_frames, n_views, j, 2), dtype=np.float32)
    confidences = np.ones((n_frames, n_views, j), dtype=np.float32)
    joints_3d = np.zeros((n_frames, j, 3), dtype=np.float32)

    for f in range(n_frames):
        X = _random_skeleton(j, rng)
        joints_3d[f] = X
        for v in range(n_views):
            points_2d[f, v] = _project(X, Ks[v], Rs[v], ts[v])

    np.savez(
        path,
        points_2d=points_2d,
        confidences=confidences,
        joints_3d=joints_3d,
        camera_K=Ks,
        camera_R=Rs,
        camera_t=ts,
    )


@pytest.mark.timeout(120)
def test_ssl_view_contrast_pretrain_cpu():
    with tempfile.TemporaryDirectory(prefix="ssl_view_contrast_smoke_") as tmp:
        tmp = Path(tmp)
        train_npz = tmp / "train.npz"
        val_npz = tmp / "val.npz"

        _write_synthetic_npz(train_npz, n_frames=80, n_views=4, j=17)
        _write_synthetic_npz(val_npz, n_frames=30, n_views=4, j=17)

        cmd = [
            sys.executable,
            str(PRETRAIN_SCRIPT),
            "--train",
            str(train_npz),
            "--val",
            str(val_npz),
            "--clip_len",
            "9",
            "--d",
            "32",
            "--residual_hidden",
            "64",
            "--n_st_layers",
            "1",
            "--epochs",
            "1",
            "--batch_size",
            "2",
            "--train_samples",
            "20",
            "--val_stride",
            "1",
            "--mask_ratio",
            "0.25",
            "--lambda_contrast",
            "0.1",
            "--contrastive_dim",
            "16",
            "--output",
            str(tmp / "ray_attention_ssl_view_contrast_smoke.pth"),
        ]

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(cmd, check=True, env=env)
