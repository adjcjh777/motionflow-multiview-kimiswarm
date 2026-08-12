"""CPU smoke test for the P18 cross-dataset benchmark manifest.

Validates that the YAML manifest is well-formed, that every referenced
``.npz`` file contains the canonical WebBridge keys, and that the
17-joint mixed loader can re-index / pad every dataset correctly.

Usage
-----
    python experiments/prototypes/swarm_iter18/P18_cross_dataset_smoke.py \
        --manifest configs/deprecated/circular/experiments/prototypes/swarm_iter18/P18_cross_dataset_manifest.yaml

Exit codes
----------
    0  all present datasets passed validation
    1  at least one present dataset failed validation or the manifest is invalid
"""

import argparse
import os
import sys
from pathlib import Path

# Avoid OpenMP duplicate-library warnings on Windows toolchains.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import yaml

# Make motionflow_mv imports available when running from repo root.
ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from motionflow_mv.data.webbridge_mixed_dataset import (  # noqa: E402
    WebBridgeCanonical17Dataset,
    DATASET_IDS,
)


REQUIRED_NPZ_KEYS = {
    "points_2d",
    "confidences",
    "joints_3d",
    "camera_K",
    "camera_R",
    "camera_t",
}


def infer_dataset_name(entry_name: str) -> str:
    """Map a manifest entry name to the loader's dataset tag."""
    name = entry_name.lower()
    for tag in ("mpi", "h36m", "aist", "shelf", "campus"):
        if tag in name:
            return tag
    raise ValueError(f"Cannot infer dataset tag for '{entry_name}'")


def validate_manifest(manifest: dict) -> None:
    """Basic structural validation of the benchmark manifest."""
    if not isinstance(manifest, dict):
        raise ValueError("Manifest YAML must contain a top-level dictionary")

    model_config = manifest.get("model_config", {})
    required_model_keys = {"model", "checkpoint", "clip_len", "d", "residual_hidden"}
    missing = required_model_keys - set(model_config.keys())
    if missing:
        raise ValueError(f"model_config missing keys: {missing}")

    datasets = manifest.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets listed in manifest")

    for entry in datasets:
        if "name" not in entry or "path" not in entry:
            raise ValueError(f"Each dataset entry needs 'name' and 'path': {entry}")


def check_npz(path: Path, dataset_name: str, clip_len: int) -> bool:
    """Load an .npz, verify keys, and exercise the 17-joint loader."""
    print(f"  Checking {path} ...")
    data = np.load(path)

    missing = REQUIRED_NPZ_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys in {path}: {missing}")

    points_2d = data["points_2d"]
    joints_3d = data["joints_3d"]
    camera_K = data["camera_K"]
    camera_R = data["camera_R"]
    camera_t = data["camera_t"]

    T, V, J, _ = points_2d.shape
    print(f"    Frames={T}, Views={V}, Joints={J}")
    print(f"    camera_K={camera_K.shape}, camera_R={camera_R.shape}, camera_t={camera_t.shape}")

    # Exercise the common-17 mixed loader (runtime re-index + view padding).
    ds = WebBridgeCanonical17Dataset(str(path), dataset_name, clip_len=clip_len)
    x, y, K, R, t, dsid = ds[0]
    print(f"    Loader output: x={tuple(x.shape)}, y={tuple(y.shape)}, "
          f"K={tuple(K.shape)}, R={tuple(R.shape)}, t={tuple(t.shape)}, "
          f"dataset_id={dsid}")

    expected_dsid = DATASET_IDS[dataset_name]
    if dsid != expected_dsid:
        raise ValueError(f"Expected dataset_id={expected_dsid}, got {dsid.item()}")

    # The mixed loader always pads to MAX_VIEWS (14) and common joints (17).
    if x.shape != (clip_len, 14, 17, 3):
        raise ValueError(f"Unexpected loader x shape {tuple(x.shape)}")
    if y.shape != (clip_len, 17, 3):
        raise ValueError(f"Unexpected loader y shape {tuple(y.shape)}")

    return True


def main():
    parser = argparse.ArgumentParser(description="P18 cross-dataset benchmark smoke test")
    parser.add_argument(
        "--manifest",
        type=str,
        default="configs/deprecated/circular/experiments/prototypes/swarm_iter18/P18_cross_dataset_manifest.yaml",
        help="Path to the benchmark YAML manifest",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        default=True,
        help="Skip missing .npz files instead of failing",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)

    clip_len = int(manifest["model_config"].get("clip_len", 13))
    print(f"Manifest: {manifest_path}")
    print(f"model_config: {manifest['model_config']['model']} "
          f"(clip_len={clip_len})")
    print(f"Torch version: {torch.__version__}")

    failures = 0
    present = 0
    skipped = 0

    for entry in manifest["datasets"]:
        name = entry["name"]
        path = Path(entry["path"])
        print(f"\n[{name}]")
        if not path.exists():
            if args.skip_missing:
                print(f"  [SKIP] not found: {path}")
                skipped += 1
                continue
            raise FileNotFoundError(f"Dataset not found: {path}")

        dataset_tag = infer_dataset_name(name)
        try:
            check_npz(path, dataset_tag, clip_len)
            present += 1
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            failures += 1

    print(f"\n{'='*60}")
    print(f"Validated: {present}  Skipped (missing): {skipped}  Failed: {failures}")
    if failures:
        print("Smoke test FAILED.")
        sys.exit(1)
    print("Smoke test PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
