#!/usr/bin/env python3
"""Convert a project-format MPI-INF-3DHP predictions .npz into an official submission package.

Input
-----
A ``.npz`` file containing one array per test sequence keyed
``TS1_v14_multiview`` .. ``TS6_v14_multiview``.
Each array has shape ``(T, 17, 3)`` and units are **metres**.

Output
------
A zip file with the per-sequence ``TS{i}.mat`` files expected by the
MPI-INF-3DHP evaluation server.  Each ``.mat`` contains ``joint_3d_image``
of shape ``(T, 17, 3)`` in **millimetres**, root-relative to the pelvis
(joint 14, 0-based).

Optionally, if ``--test_root`` is supplied, the zip also contains
``mpii_3dhp_prediction.mat`` for local evaluation with the official
test-util MATLAB scripts.

No network upload is performed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io as sio


PELVIS_IDX = 14  # 0-based pelvis in the 17-joint "relevant" skeleton
SEQUENCE_KEYS = [f"TS{i}_v14_multiview" for i in range(1, 7)]


def _import_package_submission():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "package_mpiinf3dhp_server_submission",
        Path(__file__).resolve().parents[1] / "scripts" / "package_mpiinf3dhp_server_submission.py",
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.package_submission


def load_predictions(npz_path: Path) -> dict[str, np.ndarray]:
    """Load and validate the project-format predictions .npz."""
    data = np.load(npz_path)
    missing = [k for k in SEQUENCE_KEYS if k not in data]
    if missing:
        raise ValueError(f"Missing keys in predictions npz: {missing}. Found: {list(data.files)}")

    predictions: dict[str, np.ndarray] = {}
    for key in SEQUENCE_KEYS:
        arr = np.asarray(data[key])
        if arr.ndim != 3 or arr.shape[1:] != (17, 3):
            raise ValueError(f"{key} has shape {arr.shape}; expected (T, 17, 3)")
        predictions[key] = arr
    return predictions


def root_relative_to_pelvis(coords_mm: np.ndarray) -> np.ndarray:
    pelvis = coords_mm[..., PELVIS_IDX : PELVIS_IDX + 1, :]
    return coords_mm - pelvis


def write_per_sequence_mats(predictions_mm: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, key in enumerate(SEQUENCE_KEYS, start=1):
        preds_mm = predictions_mm[key]
        preds_rel = root_relative_to_pelvis(preds_mm)
        mat_path = out_dir / f"TS{i}.mat"
        sio.savemat(
            mat_path,
            {
                "joint_3d_image": preds_rel.astype(np.float64),
                "joint_3d_image_abs": preds_mm.astype(np.float64),
                "pelvis_index": int(PELVIS_IDX),
                "unit": "mm",
            },
            do_compression=True,
        )
        written.append(mat_path)
    return written


def write_local_eval_mat(
    predictions_m: dict[str, np.ndarray],
    test_root: Path,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_npz = out_dir / ".tmp_predictions_for_local_eval.npz"
    np.savez_compressed(temp_npz, **predictions_m)

    local_mat = out_dir / "mpii_3dhp_prediction.mat"
    package_submission = _import_package_submission()
    package_submission(str(temp_npz), str(test_root), str(local_mat))

    temp_npz.unlink(missing_ok=True)
    return local_mat


def write_manifest(
    out_dir: Path,
    predictions_mm: dict[str, np.ndarray],
    method_name: str,
    source_npz: Path,
    test_root: Path | None,
) -> Path:
    manifest = {
        "method": method_name,
        "source_npz": str(source_npz),
        "test_root": str(test_root) if test_root else None,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "unit": "mm",
        "root_relative_to_pelvis_index": PELVIS_IDX,
        "variable_name": "joint_3d_image",
        "sequences": {
            key: {
                "frames": int(predictions_mm[key].shape[0]),
                "joints": int(predictions_mm[key].shape[1]),
                "mat_file": f"TS{i}.mat",
            }
            for i, key in enumerate(SEQUENCE_KEYS, start=1)
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def write_readme(out_dir: Path) -> Path:
    readme = out_dir / "README.txt"
    text = (
        "MPI-INF-3DHP official test-set submission package\n"
        "====================================================\n\n"
        "Per-sequence predictions:\n"
        "  TS1.mat .. TS6.mat\n"
        "    variable: joint_3d_image (T, 17, 3) [mm]\n"
        "    root-relative to pelvis (joint 14 in the 17-joint relevant skeleton)\n\n"
        "If present, mpii_3dhp_prediction.mat is the local evaluation format used by\n"
        "the MPI-INF-3DHP test-util MATLAB scripts.\n\n"
        "Submission steps:\n"
        "  1. Visit https://vcai.mpi-inf.mpg.de/3dhp-dataset/\n"
        "  2. Submit the per-sequence TS{i}.mat files or the submission.zip.\n\n"
        "Do not commit this zip to version control; it contains submission outputs.\n"
    )
    readme.write_text(text, encoding="utf-8")
    return readme


def zip_staging_dir(staging_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in staging_dir.iterdir():
            if path.is_file():
                zf.write(path, arcname=path.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a project-format MPI-INF-3DHP predictions .npz into an official submission package.",
    )
    parser.add_argument("--npz", required=True, type=Path, help="Path to predictions .npz (metres, keys TS1_v14_multiview .. TS6_v14_multiview).")
    parser.add_argument("--test_root", type=Path, default=None, help="Path to official MPI test set root (contains TS1..TS6). Optional; enables local-eval mat.")
    parser.add_argument("--output", type=Path, default=Path("outputs/mpi_official_submission_from_npz.zip"), help="Destination zip file.")
    parser.add_argument("--method_name", default="MotionFlow-MultiView", help="Method name for the submission manifest.")
    args = parser.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(f"Predictions .npz not found: {args.npz}")

    # Load predictions (metres) and convert to mm for the submission format.
    predictions_m = load_predictions(args.npz)
    predictions_mm = {k: v * 1000.0 for k, v in predictions_m.items()}

    # Stage the submission package.
    staging_dir = args.output.with_suffix("")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    write_per_sequence_mats(predictions_mm, staging_dir)
    if args.test_root:
        write_local_eval_mat(predictions_m, args.test_root, staging_dir)
    write_manifest(staging_dir, predictions_mm, args.method_name, args.npz, args.test_root)
    write_readme(staging_dir)

    # Zip.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    zip_staging_dir(staging_dir, args.output)

    # Clean up staging directory.
    shutil.rmtree(staging_dir, ignore_errors=True)

    print("MPI-INF-3DHP submission package prepared")
    print(f"Method:     {args.method_name}")
    print(f"Source:     {args.npz}")
    print(f"Output zip: {args.output.resolve()}")
    print(f"Sequences:  {len(predictions_mm)}")
    for i, key in enumerate(SEQUENCE_KEYS, start=1):
        print(f"  TS{i}.mat    <- {key} ({predictions_mm[key].shape[0]} frames)")
    if args.test_root:
        print("Includes local-evaluation file: mpii_3dhp_prediction.mat")
    print("No data was uploaded automatically.")


if __name__ == "__main__":
    main()
