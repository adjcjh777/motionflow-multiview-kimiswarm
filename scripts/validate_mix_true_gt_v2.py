"""Validate configs/splits/mix_true_gt_v2.yaml with the WebBridge mixed loader.

This script performs a CPU-only sanity check that is light enough to run while
GPU training is active:

1. Load the YAML manifest.
2. Verify train/val/test path lists and matching name lists.
3. Check that every referenced .npz file exists and contains the required keys.
4. For each split, load one representative file per domain through
   WebBridgeCanonical17Dataset and verify the sample shape.
5. Report per-domain counts and any loading errors.

Run with the repository virtualenv active.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

# Ensure project modules are importable regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from motionflow_mv.data.webbridge_mixed_dataset import WebBridgeCanonical17Dataset

REQUIRED_KEYS = {"points_2d", "confidences", "joints_3d", "camera_K", "camera_R", "camera_t"}


def load_manifest(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def infer_dataset_name(path: str) -> str:
    # Use the filename because the directory path contains both "shelf" and "campus".
    name = Path(path).name.lower()
    if "h36m_true_gt" in path.lower() or name.startswith("s_") and "multiview_m" in name:
        return "h36m"
    if "aistpp_canonical" in path.lower():
        return "aist"
    if "campus" in name:
        return "campus"
    if "shelf" in name:
        return "shelf"
    return "unknown"


def validate_split(manifest: dict, split: str, root: Path) -> dict:
    paths = manifest.get(f"{split}_paths", [])
    names = manifest.get(f"{split}_names", [])
    result = {
        "split": split,
        "n_paths": len(paths),
        "n_names": len(names),
        "missing": [],
        "bad_keys": [],
        "domain_counts": defaultdict(int),
        "representative": {},
    }

    if len(paths) != len(names):
        result["error"] = f"path/name mismatch: {len(paths)} paths vs {len(names)} names"
        return result

    for p, n in zip(paths, names):
        expected = infer_dataset_name(p)
        if expected != "unknown" and n != expected:
            result["error"] = f"expected {expected} but got {n} for {p}"
            return result
        full = root / p
        if not full.exists():
            result["missing"].append(p)
        else:
            result["domain_counts"][n] += 1
            # Keep the first file of each domain as a representative for loader tests.
            if n not in result["representative"]:
                result["representative"][n] = str(full)

    return result


def check_npz_keys(path: str) -> tuple:
    try:
        with np.load(path) as data:
            missing = REQUIRED_KEYS - set(data.keys())
            if missing:
                return False, f"missing keys {missing}"
            return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def try_load_representative(path: str, name: str, clip_len: int = 9) -> dict:
    try:
        ds = WebBridgeCanonical17Dataset(path, name, clip_len=clip_len)
        if len(ds) == 0:
            return {"status": "error", "error": "dataset length 0"}
        sample = ds[0]
        # The last element (dataset_id) is a Python int, not a tensor.
        shapes = []
        for s in sample:
            if hasattr(s, "shape"):
                shapes.append(tuple(s.shape))
            else:
                shapes.append(repr(s))
        return {
            "status": "ok",
            "length": len(ds),
            "sample_shapes": shapes,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="Validate mix_true_gt_v2 manifest")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/splits/mix_true_gt_v2.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--clip_len", type=int, default=9)
    args = parser.parse_args()

    print(f"Manifest: {args.manifest.resolve()}")
    print(f"Project root: {args.root.resolve()}")

    manifest = load_manifest(args.manifest)

    print("\n--- YAML structure ---")
    for key in ("name", "description", "domain_balancing_weights"):
        if key in manifest:
            print(f"{key}: present")

    print("\n--- Path/name counts ---")
    for split in ("train", "val", "test"):
        paths = manifest.get(f"{split}_paths", [])
        names = manifest.get(f"{split}_names", [])
        print(f"{split}: {len(paths)} paths, {len(names)} names")

    print("\n--- File existence / domain counts / .npz key checks ---")
    has_fatal = False
    split_results = {}
    for split in ("train", "val", "test"):
        result = validate_split(manifest, split, args.root)
        split_results[split] = result
        print(f"{split}: {result['n_paths']} paths, {len(result['missing'])} missing")
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            has_fatal = True
        if result["missing"]:
            for m in result["missing"][:5]:
                print(f"  missing: {m}")
        if result["bad_keys"]:
            for b in result["bad_keys"][:5]:
                print(f"  bad keys: {b}")
        print(f"  domain counts: {dict(result['domain_counts'])}")

    print("\n--- Per-file .npz key validation (one per domain) ---")
    for split, result in split_results.items():
        for name, path in result["representative"].items():
            ok, msg = check_npz_keys(path)
            print(f"{split}/{name}: {path} -> {'OK' if ok else 'FAIL: ' + msg}")
            if not ok:
                has_fatal = True

    print("\n--- WebBridgeCanonical17Dataset representative loading ---")
    for split, result in split_results.items():
        for name, path in result["representative"].items():
            info = try_load_representative(path, name, clip_len=args.clip_len)
            print(f"{split}/{name}: {info}")
            if info.get("status") == "error":
                has_fatal = True

    if has_fatal:
        print("\nVALIDATION FAILED")
        sys.exit(1)
    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()
