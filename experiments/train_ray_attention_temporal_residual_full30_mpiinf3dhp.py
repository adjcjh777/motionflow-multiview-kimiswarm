"""Full 30-epoch training wrapper for RayAttentionFusionModelTemporalResidual.

Launches ``experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py``
with the standard MPI-INF-3DHP cross-subject split and saves the checkpoint to
``outputs/ray_attention_temporal_residual_full30.pth``.

    Train : S1 (Seq1 + Seq1_02 + Seq2), S3 (Seq1 + Seq2)
    Val   : S2 Seq1
"""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).parent.parent
    train_script = root / "experiments" / "train_ray_attention_temporal_residual_mpiinf3dhp.py"
    data_dir = root / "data" / "webbridge" / "mpi_inf_3dhp"

    train_files = [
        data_dir / "s_01_seq_01_v14_multiview_m.npz",
        data_dir / "s_01_seq_01_02_v14_multiview_m.npz",
        data_dir / "s_01_seq_02_v14_multiview_m.npz",
        data_dir / "s_03_seq_01_v14_multiview_m.npz",
        data_dir / "s_03_seq_02_v14_multiview_m.npz",
    ]
    val_file = data_dir / "s_02_seq_01_v14_multiview_m.npz"

    missing = [p for p in train_files + [val_file] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing MPI-INF-3DHP data files: {missing}")

    cmd = [
        sys.executable,
        str(train_script),
        "--train",
        *[str(p) for p in train_files],
        "--val",
        str(val_file),
        "--clip_len", "13",
        "--d", "64",
        "--n_temporal_layers", "2",
        "--residual_hidden", "128",
        "--epochs", "30",
        "--lr", "1e-3",
        "--batch_size", "8",
        "--train_samples", "250",
        "--output",
        str(root / "outputs" / "ray_attention_temporal_residual_full30.pth"),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
