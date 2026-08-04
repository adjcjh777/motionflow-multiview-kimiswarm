"""Evaluate all registered FusionModule plugins on synthetic 3D multi-view data.

This script demonstrates that DLT, attention, robust triangulation, residual
refiner, and temporal refiner can all be run through the same plugin
interface. It reports MPJPE and PA-MPJPE on a fixed synthetic sequence.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/eval_all_plugins_synthetic.py
"""

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_attention_fusion import make_cameras
from motionflow_mv.eval.metrics import mpjpe, pa_mpjpe
from motionflow_mv.fusion import FUSION_REGISTRY


def generate_sequence(n_views: int = 5, j: int = 17, t: int = 30, seed: int = 2025):
    """Generate a synthetic multi-view sequence with smooth 3D motion."""
    rng = np.random.default_rng(seed)
    # Random skeleton that moves smoothly over time.
    base = rng.uniform(-1.0, 1.0, size=(j, 3)) * np.array([0.5, 0.8, 1.5])
    base[:, 2] += 3.0
    trajectory = np.cumsum(rng.normal(0, 0.05, size=(t, 3)), axis=0)
    joints_3d = base[None, :, :] + trajectory[:, None, :]  # (T, J, 3)

    cameras = make_cameras(n_views, rng)
    proj_matrices = np.stack([cam.projection_matrix for cam in cameras], axis=0)

    points_2d = np.zeros((t, n_views, j, 2), dtype=np.float32)
    confidences = np.ones((t, n_views, j), dtype=np.float32) * 0.9
    for v in range(n_views):
        P = proj_matrices[v]
        X_h = np.concatenate([joints_3d, np.ones((t, j, 1))], axis=-1)
        x_h = (P @ X_h.reshape(-1, 4).T).T.reshape(t, j, 3)
        points_2d[:, v] = x_h[..., :2] / x_h[..., 2:]
        points_2d[:, v] += rng.normal(0, 0.5, size=(t, j, 2))

    return points_2d, confidences, cameras, joints_3d


def main():
    n_views, j, t = 5, 17, 30
    points_2d, confidences, cameras, joints_3d_gt = generate_sequence(n_views, j, t)

    # Optionally load trained attention plugin.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path("outputs") / "attention_fusion_synthetic_plugin.pth"
    if checkpoint_path.exists():
        module = FUSION_REGISTRY.get("attention")
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        module.model.load_state_dict(state)
        module.model.to(device)
        module.model.eval()

    print("Plugin comparison on synthetic 3D GT")
    print(f"{'Plugin':<20} {'MPJPE':>10} {'PA-MPJPE':>10}")
    print("-" * 42)

    for name in sorted(FUSION_REGISTRY.names()):
        module = FUSION_REGISTRY.get(name)
        pred = module.fuse(points_2d, confidences, cameras)
        m = mpjpe(pred, joints_3d_gt)
        pa = pa_mpjpe(pred.reshape(-1, 3), joints_3d_gt.reshape(-1, 3))
        print(f"{name:<20} {m:>10.4f} {pa:>10.4f}")


if __name__ == "__main__":
    main()
