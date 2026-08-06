"""Minimal ablation runner for the Bayesian triangulation model."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINER = ROOT / "experiments" / "train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py"

VARIANTS = {
    "full": {"use_adaptive_gn": "1", "anisotropic_covariance": "1", "epipolar_loss_weight": "0.05"},
    "no_adaptive_gn": {"use_adaptive_gn": "0", "anisotropic_covariance": "1", "epipolar_loss_weight": "0.05"},
    "isotropic_cov": {"use_adaptive_gn": "1", "anisotropic_covariance": "0", "epipolar_loss_weight": "0.05"},
    "no_epipolar": {"use_adaptive_gn": "1", "anisotropic_covariance": "1", "epipolar_loss_weight": "0.0"},
}


def main():
    parser = argparse.ArgumentParser(description="Ablation study for Bayesian triangulation components")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", type=str, nargs="+", default=[
        "data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz",
        "data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m_smoke.npz",
    ])
    parser.add_argument("--val", type=str, default="data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--residual_hidden", type=int, default=64)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--val_stride", type=int, default=50)
    parser.add_argument("--variant", type=str, default="full", choices=list(VARIANTS.keys()))
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 2
        args.train_samples = 100
        args.batch_size = 2
        print("Smoke mode enabled: 2 epochs, 100 samples, batch_size=2")

    info = VARIANTS[args.variant]
    train_files = [str(ROOT / p) for p in args.train]
    val_file = str(ROOT / args.val)
    output = ROOT / "outputs" / "ablate_bayesian_tri_components" / f"{args.variant}.pth"
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(TRAINER),
        "--train", *train_files, "--val", val_file,
        "--model_type", "bayesian_tri",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--train_samples", str(args.train_samples),
        "--d", str(args.d),
        "--residual_hidden", str(args.residual_hidden),
        "--n_st_layers", str(args.n_st_layers),
        "--val_stride", str(args.val_stride),
        "--pp_loss_weight", "0.1",
        "--epipolar_loss_weight", info["epipolar_loss_weight"],
        "--use_adaptive_gn", info["use_adaptive_gn"],
        "--anisotropic_covariance", info["anisotropic_covariance"],
        "--cam_aug_pp", "5.0", "--cam_aug_focal", "0.01",
        "--cam_aug_schedule", "intrinsics_curriculum",
        "--cam_aug_intrinsics_ramp_epochs", "5",
        "--output", str(output),
    ]

    print(f"Running variant: {args.variant}")
    subprocess.run(cmd, check=True)
    print(f"Checkpoint saved to {output}")


if __name__ == "__main__":
    main()
