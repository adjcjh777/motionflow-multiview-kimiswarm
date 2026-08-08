"""Fast CPU smoke: validate the expanded WebBridge mixed manifest.

Loads ``configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml``,
cis-checks every referenced .npz, builds the mixed loader, and prints a
single-batch shape check.  This does NOT train a model, so it is safe to
run while the local 4090 v25 baseline is using the GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from motionflow_mv.data.webbridge_mixed_dataset import build_webbridge_mixed_dataloaders


MANIFEST = Path("configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml")


def main():
    with open(MANIFEST) as f:
        cfg = yaml.safe_load(f)

    train_paths = cfg["train_paths"]
    train_names = cfg["train_names"]
    val_paths = cfg["val_paths"]
    val_names = cfg["val_names"]

    missing = [p for p in train_paths + val_paths if not Path(p).exists()]
    if missing:
        print("MISSING FILES:")
        for p in missing:
            print(" ", p)
        raise SystemExit(1)

    train_loader, val_loader = build_webbridge_mixed_dataloaders(
        train_paths=train_paths,
        train_names=train_names,
        val_paths=val_paths,
        val_names=val_names,
        clip_len=13,
        batch_size=4,
        train_samples=100,
        val_stride=100,
        num_workers=0,
    )

    train_batches = len(train_loader)
    val_batches = len(val_loader)

    x, y, K, R, t, dataset_ids = next(iter(train_loader))

    print("Expanded manifest:", MANIFEST)
    print(f"  train files: {len(train_paths)}  val files: {len(val_paths)}")
    print(f"  train batches: {train_batches}  val batches: {val_batches}")
    print(f"  sample x shape: {tuple(x.shape)} (B,T,V,J,3)")
    print(f"  sample y shape: {tuple(y.shape)} (B,T,J,3)")
    print(f"  camera K shape: {tuple(K.shape)}")
    print(f"  dataset_ids: {dataset_ids.tolist()}")
    print("Loader smoke: OK")


if __name__ == "__main__":
    main()
