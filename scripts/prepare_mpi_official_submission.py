#!/usr/bin/env python3
"""Prepare a zip package for the official MPI-INF-3DHP test-server submission.

The script supports three input modes:

1. **Existing predictions .npz** (``--mode predictions --npz <path>``): the .npz
   must contain one array per test sequence with keys ``TS1_v14_multiview`` ..
   ``TS6_v14_multiview``, each of shape ``(T, 17, 3)`` in metres.
2. **DLT triangulation of the canonical test-set .npz files**
   (``--mode dlt --test_set_dir <dir>``): triangulates the stored 2D keypoints
   using the stored calibrated cameras.  This only works if the .npz files
   contain 2D points for at least two views per frame (the official release only
   contains one annotated view, so detected-2D for the remaining views is
   required).
3. **Trained OmniMultiViewFusionV2 checkpoint**
   (``--mode checkpoint --checkpoint <path> --test_set_dir <dir>``): runs the
   existing inference script, then packages the resulting .npz.

The output zip contains, for each test sequence, a ``TS{i}.mat`` file with the
predicted 3D joints in millimetres, root-relative to the pelvis (joint 14 in the
17-joint "relevant" skeleton).  The prediction variable is named
``joint_3d_image`` and has shape ``(T, 17, 3)``.

Optionally, if the official test-set root (with ``TS{i}/annot_data.mat``) is
provided, the script also produces ``mpii_3dhp_prediction.mat`` in the zip.  That
file follows the local evaluation format used by the MPI-INF-3DHP test-util
scripts (``sequencewise_per_joint_error`` and ``sequencewise_activity_labels``).

No network upload is performed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io as sio


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PELVIS_IDX = 14  # 0-based pelvis in the MPI-INF-3DHP 17-joint "relevant" skeleton
SEQUENCE_KEYS = [f"TS{i}_v14_multiview" for i in range(1, 7)]


# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

# Import the existing local-evaluation packager on demand.
sys.path.insert(0, str(REPO_ROOT))
from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq  # noqa: E402


def _import_package_submission():
    """Import the helper that builds the local-evaluation .mat."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "package_mpiinf3dhp_server_submission",
        REPO_ROOT / "scripts" / "package_mpiinf3dhp_server_submission.py",
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.package_submission


# ---------------------------------------------------------------------------
# DLT / inference helpers
# ---------------------------------------------------------------------------

def _build_projection_matrices(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return (V, 3, 4) projection matrices P = K [R | t]."""
    V = K.shape[0]
    P = np.zeros((V, 3, 4), dtype=np.float64)
    for v in range(V):
        Rt = np.concatenate([R[v], t[v].reshape(3, 1)], axis=1)
        P[v] = K[v] @ Rt
    return P


def triangulate_npz_sequence(npz_path: Path, device: str = "cpu") -> np.ndarray:
    """Run confidence-weighted DLT on a single canonical .npz and return (T, J, 3) in metres.

    Raises:
        ValueError: if fewer than two views are available for triangulation.
    """
    data = np.load(npz_path)

    points_2d = np.asarray(data["points_2d"], dtype=np.float64)
    confidences = np.asarray(data["confidences"], dtype=np.float64)
    K = np.asarray(data["camera_K"], dtype=np.float64)
    R = np.asarray(data["camera_R"], dtype=np.float64)
    t = np.asarray(data["camera_t"], dtype=np.float64)

    if points_2d.ndim != 4:
        raise ValueError(f"Expected points_2d shape (T,V,J,2), got {points_2d.shape} in {npz_path}")

    T, V, J, _ = points_2d.shape

    # Determine how many views are active per frame.
    active_per_frame = (confidences.sum(axis=2) > 0).sum(axis=1)  # (T,)
    if active_per_frame.mean() < 2:
        raise ValueError(
            f"{npz_path.name}: only {active_per_frame.mean():.1f} active views on average. "
            "DLT requires at least 2 views. The official test-set .npz files only contain "
            "one annotated view; run a multi-view 2D detector first."
        )

    P = _build_projection_matrices(K, R, t)
    P_t = np.tile(P[None], (T, 1, 1, 1))

    p2d_t = np.asarray(points_2d, dtype=np.float64)
    w_t = np.asarray(confidences, dtype=np.float64)

    # Use torch only if a CUDA device is requested; otherwise keep it lightweight on CPU.
    if device == "cpu":
        import torch

        p2d_torch = torch.from_numpy(p2d_t)
        P_torch = torch.from_numpy(P_t)
        w_torch = torch.from_numpy(w_t)
        X_t = triangulate_dlt_batched_lstsq(p2d_torch, P_torch, weights=w_torch)
        X = X_t.detach().cpu().numpy()
    else:
        import torch

        p2d_torch = torch.from_numpy(p2d_t).to(device=device, dtype=torch.float64)
        P_torch = torch.from_numpy(P_t).to(device=device, dtype=torch.float64)
        w_torch = torch.from_numpy(w_t).to(device=device, dtype=torch.float64)
        X_t = triangulate_dlt_batched_lstsq(p2d_torch, P_torch, weights=w_torch)
        X = X_t.detach().cpu().numpy()

    return X.reshape(T, J, 3)


def run_inference(checkpoint: Path, test_set_dir: Path, out_npz: Path, device: str) -> None:
    """Run the OmniMultiViewFusionV2 inference script to produce a predictions .npz."""
    script = REPO_ROOT / "experiments" / "infer_mpiinf3dhp_test_set_omniview_v2.py"
    if not script.exists():
        raise FileNotFoundError(f"Inference script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--checkpoint",
        str(checkpoint),
        "--test_set_dir",
        str(test_set_dir),
        "--out_npz",
        str(out_npz),
        "--device",
        device,
    ]
    subprocess.run(cmd, check=True)


def load_predictions_from_npz(npz_path: Path) -> dict[str, np.ndarray]:
    """Load predictions .npz and validate required keys."""
    data = np.load(npz_path)
    missing = [k for k in SEQUENCE_KEYS if k not in data]
    if missing:
        raise ValueError(f"Predictions .npz missing keys: {missing}. Found: {list(data.files)}")

    predictions: dict[str, np.ndarray] = {}
    for key in SEQUENCE_KEYS:
        arr = np.asarray(data[key])
        if arr.ndim != 3 or arr.shape[1:] != (17, 3):
            raise ValueError(f"{key} has shape {arr.shape}; expected (T, 17, 3)")
        predictions[key] = arr
    return predictions


def dlt_predictions(test_set_dir: Path, device: str = "cpu") -> dict[str, np.ndarray]:
    """Triangulate every TS{i}_v14_multiview.npz in ``test_set_dir``."""
    predictions: dict[str, np.ndarray] = {}
    for key in SEQUENCE_KEYS:
        npz_path = Path(test_set_dir) / f"{key}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Test-set file not found: {npz_path}")
        print(f"DLT triangulating {npz_path.name} ...")
        preds = triangulate_npz_sequence(npz_path, device=device)
        predictions[key] = preds
        print(f"  -> {preds.shape[0]} frames, {preds.shape[1]} joints")
    return predictions


def root_relative_to_pelvis(coords_mm: np.ndarray) -> np.ndarray:
    """Subtract the pelvis joint from all joints.

    Parameters
    ----------
    coords_mm: (..., J, 3) array in millimetres.

    Returns
    -------
    Root-relative (..., J, 3) array.
    """
    pelvis = coords_mm[..., PELVIS_IDX : PELVIS_IDX + 1, :]
    return coords_mm - pelvis


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_per_sequence_mats(
    predictions_mm: dict[str, np.ndarray], out_dir: Path
) -> list[Path]:
    """Write ``TS{i}.mat`` files containing ``joint_3d_image`` variables."""
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
    predictions: dict[str, np.ndarray], test_root: Path, out_dir: Path
) -> Path:
    """Produce ``mpii_3dhp_prediction.mat`` for local evaluation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_npz = out_dir / ".tmp_predictions_for_local_eval.npz"
    np.savez_compressed(temp_npz, **predictions)

    local_mat = out_dir / "mpii_3dhp_prediction.mat"
    package_submission = _import_package_submission()
    package_submission(str(temp_npz), str(test_root), str(local_mat))

    temp_npz.unlink(missing_ok=True)
    return local_mat


def write_manifest(
    out_dir: Path,
    predictions: dict[str, np.ndarray],
    method_name: str,
    source_mode: str,
    test_set_dir: Path,
    test_root: Path | None,
) -> Path:
    """Write a human- and machine-readable manifest inside the package."""
    manifest = {
        "method": method_name,
        "source_mode": source_mode,
        "test_set_dir": str(test_set_dir),
        "test_root": str(test_root) if test_root else None,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "unit": "mm",
        "root_relative_to_pelvis_index": PELVIS_IDX,
        "variable_name": "joint_3d_image",
        "sequences": {
            key: {
                "frames": int(predictions[key].shape[0]),
                "joints": int(predictions[key].shape[1]),
                "mat_file": f"TS{i}.mat",
            }
            for i, key in enumerate(SEQUENCE_KEYS, start=1)
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def write_readme(out_dir: Path) -> Path:
    """Write a short README explaining the package contents."""
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
        "See docs/mpi_official_submission_protocol.md for submission instructions.\n\n"
        "Do not commit this zip to version control; it contains submission outputs.\n"
    )
    readme.write_text(text, encoding="utf-8")
    return readme


def zip_staging_dir(staging_dir: Path, zip_path: Path) -> None:
    """Zip every file in ``staging_dir`` into ``zip_path``."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in staging_dir.iterdir():
            if path.is_file():
                zf.write(path, arcname=path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a zip package for the MPI-INF-3DHP official test server.",
    )
    parser.add_argument(
        "--mode",
        choices=["predictions", "dlt", "checkpoint"],
        required=True,
        help=(
            "Input mode: 'predictions' uses an existing .npz; 'dlt' triangulates the "
            "canonical test-set .npz files; 'checkpoint' runs OmniMultiViewFusionV2 inference."
        ),
    )
    parser.add_argument(
        "--npz",
        type=Path,
        help="Path to an existing predictions .npz (required for --mode predictions).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to a trained OmniMultiViewFusionV2 checkpoint (required for --mode checkpoint).",
    )
    parser.add_argument(
        "--test_set_dir",
        type=Path,
        default=Path("data/webbridge/mpi_inf_3dhp/test_set"),
        help="Directory containing TS{i}_v14_multiview.npz files.",
    )
    parser.add_argument(
        "--test_root",
        type=Path,
        default=None,
        help=(
            "Path to the downloaded MPI-INF-3DHP test set root (containing TS1..TS6). "
            "If provided, a local-evaluation mat (mpii_3dhp_prediction.mat) is included in the zip."
        ),
    )
    parser.add_argument(
        "--output_zip",
        type=Path,
        default=Path("outputs/mpi_official_submission.zip"),
        help="Destination zip file.",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="MotionFlow-MultiView",
        help="Method name to record in the submission manifest.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device for DLT/inference (cpu or cuda).",
    )

    args = parser.parse_args()

    if args.mode == "predictions" and not args.npz:
        parser.error("--npz is required when --mode predictions")
    if args.mode == "checkpoint" and not args.checkpoint:
        parser.error("--checkpoint is required when --mode checkpoint")

    return args


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Gather predictions (metres).
    # ------------------------------------------------------------------
    if args.mode == "predictions":
        print(f"Loading predictions from {args.npz}")
        predictions_m = load_predictions_from_npz(args.npz)
    elif args.mode == "dlt":
        predictions_m = dlt_predictions(args.test_set_dir, device=args.device)
    else:  # checkpoint
        temp_npz = Path(tempfile.gettempdir()) / f"mpi_submission_{id(object())}.npz"
        try:
            run_inference(args.checkpoint, args.test_set_dir, temp_npz, args.device)
            predictions_m = load_predictions_from_npz(temp_npz)
        finally:
            temp_npz.unlink(missing_ok=True)

    # Convert to millimetres for the submission format.
    predictions_mm = {k: v * 1000.0 for k, v in predictions_m.items()}

    # ------------------------------------------------------------------
    # Stage the submission package.
    # ------------------------------------------------------------------
    staging_dir = args.output_zip.with_suffix("")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    write_per_sequence_mats(predictions_mm, staging_dir)
    if args.test_root:
        write_local_eval_mat(predictions_m, args.test_root, staging_dir)
    write_manifest(
        staging_dir,
        predictions_mm,
        args.method_name,
        args.mode,
        args.test_set_dir,
        args.test_root,
    )
    write_readme(staging_dir)

    # ------------------------------------------------------------------
    # Zip the package.
    # ------------------------------------------------------------------
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    if args.output_zip.exists():
        args.output_zip.unlink()
    zip_staging_dir(staging_dir, args.output_zip)

    # Clean up the staging directory to keep the workspace tidy.
    shutil.rmtree(staging_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("MPI-INF-3DHP submission package prepared")
    print("=" * 70)
    print(f"Method:        {args.method_name}")
    print(f"Mode:          {args.mode}")
    print(f"Output zip:    {args.output_zip.resolve()}")
    print(f"Sequences:     {len(predictions_mm)}")
    print("Per-sequence .mat files inside the zip:")
    for i, key in enumerate(SEQUENCE_KEYS, start=1):
        print(f"  TS{i}.mat    <- {key} ({predictions_mm[key].shape[0]} frames)")
    if args.test_root:
        print("Includes local-evaluation file: mpii_3dhp_prediction.mat")
    print("No data was uploaded to the server.")


if __name__ == "__main__":
    main()
