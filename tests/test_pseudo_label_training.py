"""CPU smoke test for semi-supervised pseudo-label training.

Generates tiny synthetic labeled and unlabeled multi-view .npz files and runs
``experiments/train_pseudo_label_ray_attention_mpiinf3dhp.py`` for one epoch on
the CPU.  The test only checks that the script completes without crashing and
produces a checkpoint.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


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
    X_cam = (R @ X.T).T + t
    x = X_cam[:, :2] / (X_cam[:, 2:3] + 1e-6)
    Kx = (K @ np.hstack([x, np.ones((X.shape[0], 1))]).T).T
    return Kx[:, :2]


def _random_skeleton(j: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.normal(0.0, 0.15, size=(j, 3))
    X[:, 2] += 2.0
    return X


def _write_npz(path: Path, n_frames: int, n_views: int, j: int, labeled: bool = True):
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

    data = {
        "points_2d": points_2d,
        "confidences": confidences,
        "camera_K": Ks,
        "camera_R": Rs,
        "camera_t": ts,
    }
    if labeled:
        data["joints_3d"] = joints_3d
    np.savez(path, **data)


def main():
    root = Path(__file__).parent.parent
    script = root / "experiments" / "train_pseudo_label_ray_attention_mpiinf3dhp.py"

    with tempfile.TemporaryDirectory(prefix="pseudo_label_smoke_") as tmp:
        tmp = Path(tmp)
        labeled_npz = tmp / "labeled.npz"
        unlabeled_npz = tmp / "unlabeled.npz"
        val_npz = tmp / "val.npz"

        print("Generating synthetic labeled/unlabeled/val .npz files...")
        _write_npz(labeled_npz, n_frames=80, n_views=4, j=17, labeled=True)
        _write_npz(unlabeled_npz, n_frames=80, n_views=4, j=17, labeled=False)
        _write_npz(val_npz, n_frames=30, n_views=4, j=17, labeled=True)

        output_ckpt = tmp / "pseudo_label_smoke.pth"

        cmd = [
            sys.executable,
            str(script),
            "--train", str(labeled_npz),
            "--val", str(val_npz),
            "--unlabeled", str(unlabeled_npz),
            "--clip_len", "9",
            "--d", "32",
            "--residual_hidden", "64",
            "--n_st_layers", "1",
            "--epochs", "1",
            "--teacher_epochs", "1",
            "--batch_size", "2",
            "--train_samples", "20",
            "--pseudo_train_samples", "20",
            "--val_stride", "1",
            "--lambda_pseudo", "0.5",
            "--pseudo_conf_thresh", "5.0",
            "--output", str(output_ckpt),
        ]

        print("\n--- Running one CPU epoch of pseudo-label training ---")
        print("Command:", " ".join(str(c) for c in cmd))
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(cmd, check=True, env=env)

        assert output_ckpt.exists(), f"Expected checkpoint {output_ckpt} to be created"
        state = torch.load(output_ckpt, map_location="cpu", weights_only=True)
        assert any(k.startswith("residual_mlp") for k in state), "Checkpoint should contain model parameters"
        print("Pseudo-label training smoke test passed.")


if __name__ == "__main__":
    main()
