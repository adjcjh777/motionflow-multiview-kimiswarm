"""Compare DLT triangulation vs trained AttentionFusion on synthetic data.

Usage:
    .venv/bin/python experiments/compare_dlt_attention_synthetic.py
"""

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_attention_fusion import generate_sample
from motionflow_mv.pipeline import MultiViewPipeline
from motionflow_mv.eval.metrics import mpjpe
from motionflow_mv.fusion.attention_model import AttentionFusionModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate test sample
    rng = np.random.default_rng(2024)
    input_tensor, target_3d, cameras = generate_sample(n_views=4, j=17, rng=rng)
    points_2d = input_tensor[..., :2].numpy()
    confidences = input_tensor[..., 2].numpy()

    # DLT triangulation
    pipeline = MultiViewPipeline(estimator=None)
    pred_dlt = pipeline.fuse_frame(points_2d, confidences, cameras)
    dlt_error = mpjpe(pred_dlt, target_3d.numpy())

    # AttentionFusion prediction
    n_views = input_tensor.shape[0]
    model = AttentionFusionModel(j=17, d=32, n_views=n_views).to(device)
    checkpoint_path = Path("outputs") / "attention_fusion_synthetic.pth"
    if not checkpoint_path.exists():
        print("No trained checkpoint found. Run train_attention_fusion.py first.")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        pred_attention = model(input_tensor.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    attention_error = mpjpe(pred_attention, target_3d.numpy())

    print(f"DLT MPJPE:           {dlt_error:.4f}")
    print(f"Attention MPJPE:     {attention_error:.4f}")
    print(f"GT centroid:         {target_3d.mean(dim=0).numpy()}")
    print(f"DLT centroid:        {pred_dlt.mean(axis=0)}")
    print(f"Attention centroid:  {pred_attention.mean(axis=0)}")


if __name__ == "__main__":
    main()
