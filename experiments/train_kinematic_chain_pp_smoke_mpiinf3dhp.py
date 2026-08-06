"""Smoke trainer for the kinematic-chain graph refiner (Tier-1 iter15 proposal).

This is a thin wrapper around ``train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py``
that runs a 5-epoch smoke on a small MPI-INF-3DHP split with the kinematic-chain
graph refiner model.
"""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).parent.parent
    trainer = root / "experiments" / "train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py"

    cmd = [
        sys.executable,
        str(trainer),
        "--train",
        str(root / "data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz"),
        "--val",
        str(root / "data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz"),
        "--clip_len", "13",
        "--d", "32",
        "--residual_hidden", "64",
        "--n_st_layers", "2",
        "--model_type", "kinematic_chain",
        "--epochs", "5",
        "--train_samples", "500",
        "--batch_size", "8",
        "--val_stride", "50",
        "--pp_loss_weight", "0.1",
        "--cam_aug_pp", "5.0",
        "--cam_aug_focal", "0.01",
        "--output", "outputs/kinematic_chain_pp_smoke.pth",
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
