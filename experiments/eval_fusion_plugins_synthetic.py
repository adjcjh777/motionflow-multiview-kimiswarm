"""Evaluate FusionModule plugins on synthetic 3D multi-view data.

This script demonstrates the plugin interface: it loads a trained
AttentionFusionModel checkpoint, wraps it as a FusionModule, and compares
it with the deterministic DLTFusion baseline on synthetic 3D ground truth.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/eval_fusion_plugins_synthetic.py
"""

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_attention_fusion import generate_sample
from motionflow_mv.eval.metrics import mpjpe, pa_mpjpe
from motionflow_mv.fusion import FUSION_REGISTRY
from motionflow_mv.fusion.attention_fusion_module import AttentionFusionModule


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_views, j = 4, 17
    rng = np.random.default_rng(2025)

    # Generate a fixed synthetic test set.
    inputs, targets, cameras = generate_sample(n_views, j, rng)
    points_2d = inputs[..., :2].numpy()[None]       # (1, V, J, 2)
    confidences = inputs[..., 2].numpy()[None]      # (1, V, J)

    # DLTFusion baseline.
    dlt_module = FUSION_REGISTRY.get("dlt")
    pred_dlt = dlt_module.fuse(points_2d, confidences, cameras)

    # AttentionFusion plugin loaded from checkpoint (if available).
    attention_module = FUSION_REGISTRY.get("attention")
    checkpoint_path = Path("outputs") / "attention_fusion_synthetic.pth"
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        attention_module.model.load_state_dict(state)
        attention_module.model.to(device)
        attention_module.model.eval()
        pred_attention = attention_module.fuse(points_2d, confidences, cameras)
    else:
        print(f"Checkpoint not found at {checkpoint_path}; train with train_attention_fusion.py first.")
        pred_attention = None

    # Report metrics in the same arbitrary units as the synthetic data.
    target_np = targets.numpy()
    pred_dlt = pred_dlt.squeeze(0)
    if pred_attention is not None:
        pred_attention = pred_attention.squeeze(0)
    print("Fusion plugin evaluation on synthetic 3D GT")
    print(f"DLT      MPJPE: {mpjpe(pred_dlt, target_np):.4f}  PA-MPJPE: {pa_mpjpe(pred_dlt, target_np):.4f}")
    if pred_attention is not None:
        print(f"Attention MPJPE: {mpjpe(pred_attention, target_np):.4f}  PA-MPJPE: {pa_mpjpe(pred_attention, target_np):.4f}")


if __name__ == "__main__":
    main()
