"""Prototype launcher for Bayesian tri v2 with extended camera perturbation.

This is a drop-in wrapper around the principal-point trainer that uses the new
``extended_curriculum`` schedule and larger perturbation ranges.  Intended for the
next full-scale run after the current anchor experiment finishes; do not start it
while another GPU training job is running.
"""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).parent.parent.parent
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
        "--d", "128",
        "--residual_hidden", "256",
        "--n_st_layers", "3",
        "--model_type", "bayesian_tri_v2",
        "--epochs", "50",
        "--train_samples", "2000",
        "--batch_size", "8",
        "--val_stride", "50",
        "--pp_loss_weight", "0.2",
        "--epipolar_loss_weight", "0.05",
        "--reproj_weight", "0.0",
        "--cam_aug_rot", "2.0",
        "--cam_aug_trans", "0.02",
        "--cam_aug_focal", "0.05",
        "--cam_aug_pp", "10.0",
        "--cam_aug_schedule", "extended_curriculum",
        "--cam_aug_ramp_epochs", "10",
        "--cam_aug_intrinsics_ramp_epochs", "5",
        "--cam_aug_warmup_epochs", "2",
        "--pp_pretrain_epochs", "3",
        "--output", "outputs/bayesian_tri_v2_extended_camera_curriculum_mpiinf3dhp.pth",
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
