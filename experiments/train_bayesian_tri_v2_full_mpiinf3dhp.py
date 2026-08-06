"""Full trainer for Bayesian triangulation v2 (batched lstsq DLT).

Runs the same 20-epoch MPI-INF-3DHP protocol as the baseline Bayesian
triangulation model, but uses ``RayAttentionFusionModelBayesianTriV2`` with the
fully batched DLT head.  This is a drop-in replacement for
``experiments/train_bayesian_tri_pp_full_mpiinf3dhp.py``.
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
        str(root / "data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz"),
        str(root / "data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz"),
        str(root / "data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz"),
        "--val",
        str(root / "data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz"),
        "--clip_len", "13",
        "--d", "64",
        "--residual_hidden", "128",
        "--n_st_layers", "2",
        "--model_type", "bayesian_tri_v2",
        "--epochs", "20",
        "--train_samples", "1000",
        "--batch_size", "8",
        "--val_stride", "50",
        "--pp_loss_weight", "0.2",
        "--epipolar_loss_weight", "0.05",
        "--reproj_weight", "0.0",
        "--cam_aug_pp", "5.0",
        "--cam_aug_focal", "0.01",
        "--cam_aug_schedule", "intrinsics_curriculum",
        "--cam_aug_intrinsics_ramp_epochs", "5",
        "--pp_pretrain_epochs", "3",
        "--output", "outputs/bayesian_tri_v2_full_mpiinf3dhp.pth",
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
