"""Cross-dataset evaluation for RayAttentionFusionModelTemporalMixedResidual.

Evaluates a trained mixed-dataset checkpoint on MPI-INF-3DHP, AIST++, and Human3.6M
validation sequences.  Each validation set is routed through its corresponding
per-dataset output/residual head.

Example
-------
    conda run -n mf python experiments/eval_ray_attention_temporal_mixed_residual_v1.py \
        --checkpoint outputs/ray_attention_temporal_mixed_residual_v1.pth \
        --mpi_val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --aist_val data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR1_ch01_multiview.npz \
        --h36m_val data/h36m_hf/s_05_acts_02_multiview.npz \
        --clip_len 13 --d 32
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_mixed_residual_v1 import (
    RayAttentionFusionModelTemporalMixedResidual,
)
from experiments.train_ray_attention_temporal_mixed_residual_v1 import (
    MixedTemporalDataset,
    collate_fn,
    DATASET_IDS,
)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    for xb, yb, K, R, t, ids in loader:
        xb, yb = xb.to(device), yb.to(device)
        K, R, t = K.to(device), R.to(device), t.to(device)
        ids = ids.to(device)
        pred, mask = model(xb, K=K, R=R, t=t, dataset_ids=ids)
        err = (pred - yb).norm(dim=-1) * mask.float()  # (B, T, J)
        total_err += err.sum().item()
        total_count += mask.sum().item()
    return total_err / total_count


def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset evaluation for mixed residual model"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--mpi_val", type=str, default=None, help="MPI-INF-3DHP val .npz")
    parser.add_argument("--aist_val", type=str, default=None, help="AIST++ val .npz")
    parser.add_argument("--h36m_val", type=str, default=None, help="Human3.6M val .npz")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out", type=str, default="outputs/mixed_residual_cross_dataset_eval.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = RayAttentionFusionModelTemporalMixedResidual(
        d=args.d,
        n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    print(f"Loaded checkpoint from {args.checkpoint}")

    results = {}

    val_specs = {
        "mpi": args.mpi_val,
        "aist": args.aist_val,
        "h36m": args.h36m_val,
    }

    for dataset_name, path in val_specs.items():
        if path is None:
            continue
        print(f"\nEvaluating on {dataset_name}: {path}")
        dataset = MixedTemporalDataset(path, dataset_name, args.clip_len, n_samples=None, stride=1)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        mpjpe_m = evaluate(model, loader, device)
        mpjpe_mm = mpjpe_m * 1000.0
        results[dataset_name] = {"path": path, "mpjpe_m": mpjpe_m, "mpjpe_mm": mpjpe_mm}
        print(f"  MPJPE: {mpjpe_m:.6f} m = {mpjpe_mm:.2f} mm")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
