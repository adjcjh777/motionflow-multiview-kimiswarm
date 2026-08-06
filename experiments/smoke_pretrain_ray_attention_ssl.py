"""CPU smoke test for masked-view reprojection pre-training.

Generates a tiny synthetic multi-view ``.npz`` dataset, demonstrates the new
``MaskedViewReprojectionDataset`` data loading behaviour, and runs
``experiments/pretrain_ray_attention_ssl.py`` for one epoch on the CPU.

Usage
-----
    python experiments/smoke_pretrain_ray_attention_ssl.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


# Reuse the same coordinate helpers as smoke_reprojgate.py for consistency.
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


def _report_data_loading(train_npz: Path, val_npz: Path):
    """Inspect a few batches from the masked-view reprojection loader."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from motionflow_mv.data.ssl_dataset import make_ssl_dataloaders_with_masking

    loader, val_loader = make_ssl_dataloaders_with_masking(
        [str(train_npz)],
        str(val_npz),
        clip_len=9,
        batch_size=2,
        mask_ratio=0.25,
        mask_mode="mixed",
        train_samples=8,
        val_stride=1,
    )

    print("\n--- Masked-view reprojection data loading report ---")
    print(f"train batches: {len(loader)}, val batches: {len(val_loader)}")

    for batch_idx, (x_masked, mask, x, K, R, t) in enumerate(loader):
        if batch_idx >= 2:
            break
        total_slots = mask.numel()
        masked_slots = mask.sum().item()
        visible_slots = total_slots - masked_slots
        print(
            f"batch {batch_idx}: x_masked={list(x_masked.shape)}, "
            f"mask={list(mask.shape)}, original_conf={list(x.shape)}, "
            f"K={list(K.shape)}, R={list(R.shape)}, t={list(t.shape)}"
        )
        print(
            f"  masked slots: {masked_slots}/{total_slots} "
            f"({100.0 * masked_slots / total_slots:.1f}%), "
            f"visible slots: {visible_slots}"
        )
        print(
            f"  confidence after masking (min/mean/max): "
            f"{x_masked[..., 2].min().item():.3f}/"
            f"{x_masked[..., 2].mean().item():.3f}/"
            f"{x_masked[..., 2].max().item():.3f}"
        )
        # Sanity: masked positions should have zero confidence.
        zero_conf_mask = (x_masked[..., 2] == 0.0) & mask
        assert zero_conf_mask.float().sum().item() == masked_slots



def main():
    root = Path(__file__).parent.parent
    pretrain_script = root / "experiments" / "pretrain_ray_attention_ssl.py"

    with tempfile.TemporaryDirectory(prefix="pretrain_ssl_smoke_") as tmp:
        tmp = Path(tmp)
        train_npz = tmp / "train.npz"
        val_npz = tmp / "val.npz"

        print("Generating synthetic train/val .npz files...")
        _write_synthetic_npz(train_npz, n_frames=80, n_views=4, j=17)
        _write_synthetic_npz(val_npz, n_frames=30, n_views=4, j=17)

        _report_data_loading(train_npz, val_npz)

        cmd = [
            sys.executable,
            str(pretrain_script),
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
            "--output",
            str(tmp / "ray_attention_ssl_smoke.pth"),
        ]

        sys.stdout.flush()
        print("\n--- Running one CPU epoch of pretrain_ray_attention_ssl.py ---")
        print("Command:", " ".join(cmd))
        sys.stdout.flush()
        # Inherit the environment and hide any CUDA devices to keep the smoke
        # test on the CPU.
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(cmd, check=True, env=env)
        print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
