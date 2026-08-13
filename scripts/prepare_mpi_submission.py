#!/usr/bin/env python3
"""Prepare MPI-INF-3DHP official test-server submission from a v5 checkpoint.

This script loads a trained OmniMultiViewFusionV5 checkpoint (e.g. v25/v85/v86),
runs inference on the official MPI-INF-3DHP test set (TS1-TS6), and writes the
predictions in the format expected by the MPI-INF-3DHP evaluation server.

Output layout (per model) under ``outputs/mpi_submissions/{model_name}/``:

* ``TS1.mat`` .. ``TS6.mat`` – per-sequence predictions. Each ``.mat`` contains
  a single variable ``joint_3d_image`` of shape ``(T, 17, 3)`` in millimetres,
  root-relative to the pelvis (joint 14 of the 17-joint relevant skeleton).
* ``manifest.json`` – machine-readable description of the package.
* ``README.txt`` – short human-readable submission instructions.
* ``submission.zip`` – zip containing the six ``.mat`` files.

Optionally, if ``--test_root`` (the downloaded MPI test set root) is supplied,
a ``mpii_3dhp_prediction.mat`` is also generated for local evaluation with the
MPI test-util scripts.

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
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# We reuse the training script's model builder so that the exact same
# architecture kwargs are available.  ``parse_args`` is called with a patched
# ``sys.argv`` to avoid picking up this script's command-line arguments.
from experiments.train_omniview_fusion_v5_webbridge_multi import (  # noqa: E402
    build_model_from_args,
    parse_args as _train_parse_args,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PELVIS_IDX = 14  # 0-based pelvis in the MPI-INF-3DHP 17-joint relevant skeleton
SEQUENCE_KEYS = [f"TS{i}_v14_multiview" for i in range(1, 7)]


# ---------------------------------------------------------------------------
# Args / config
# ---------------------------------------------------------------------------

def _train_args_with_defaults() -> argparse.Namespace:
    """Return the training script's default Namespace without parsing sys.argv."""
    old_argv = sys.argv
    sys.argv = ["train_omniview_fusion_v5_webbridge_multi.py"]
    try:
        return _train_parse_args()
    finally:
        sys.argv = old_argv


def namespace_from_yaml(config_path: Path) -> argparse.Namespace:
    """Build an argparse Namespace from a YAML training config.

    The YAML file is expected to contain a ``training:`` section whose keys map
    1:1 to the argparse flags of ``train_omniview_fusion_v5_webbridge_multi.py``.
    Any missing flags keep the training script's defaults.
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    args = _train_args_with_defaults()

    training = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    if not isinstance(training, dict):
        raise ValueError(f"YAML config {config_path} must contain a 'training' mapping")

    for key, value in training.items():
        if not hasattr(args, key):
            raise ValueError(
                f"Unknown key '{key}' in YAML config {config_path}. "
                "This key does not correspond to a known training flag."
            )
        setattr(args, key, value)

    return args


# ---------------------------------------------------------------------------
# Dataset / inference
# ---------------------------------------------------------------------------

class TemporalClipDataset(torch.utils.data.Dataset):
    """Yield overlapping temporal clips from a canonical multi-view .npz."""

    def __init__(self, npz_path: str, clip_len: int, stride: int = 1):
        data = np.load(npz_path)

        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()

        self.clip_len = clip_len
        self.stride = stride
        self.total_frames = self.points_2d.shape[0]
        self.num_clips = max(1, (self.total_frames - clip_len) // stride + 1)

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        return x, self.K, self.R, self.t, start


def collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    x = torch.stack([b[0] for b in batch], dim=0)
    K = torch.stack([b[1] for b in batch], dim=0)
    R = torch.stack([b[2] for b in batch], dim=0)
    t = torch.stack([b[3] for b in batch], dim=0)
    starts = np.array([b[4] for b in batch], dtype=np.int64)
    return x, K, R, t, starts


def infer_sequence(
    model: torch.nn.Module,
    npz_path: str,
    clip_len: int,
    batch_size: int,
    stride: int,
    device: torch.device,
) -> np.ndarray:
    """Run sliding-window inference on one test sequence.

    Returns
    -------
    Per-frame 3D poses as a NumPy array of shape ``(T, J, 3)`` in metres.
    """
    dataset = TemporalClipDataset(npz_path, clip_len, stride=stride)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    total_frames = dataset.total_frames
    n_joints = dataset.points_2d.shape[2]
    accum = torch.zeros((total_frames, n_joints, 3), dtype=torch.float32)
    counts = torch.zeros((total_frames, n_joints, 1), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        for xb, K, R, t, starts in loader:
            xb = xb.to(device)
            K = K.to(device)
            R = R.to(device)
            t = t.to(device)
            out = model(xb, K=K, R=R, t=t)
            pred = out[0].cpu()

            B, T, J, _ = pred.shape
            for i in range(B):
                start = int(starts[i])
                end = start + T
                accum[start:end] += pred[i]
                counts[start:end] += 1.0

    counts = counts.clamp(min=1.0)
    per_frame = accum / counts
    return per_frame.numpy()


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Checkpoint load: missing keys {missing[:10]}")
    if unexpected:
        print(f"Checkpoint load: unexpected keys ignored {unexpected[:10]}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def root_relative_to_pelvis(coords_mm: np.ndarray) -> np.ndarray:
    """Subtract the pelvis joint from all joints.

    Parameters
    ----------
    coords_mm: array of shape ``(..., J, 3)`` in millimetres.

    Returns
    -------
    Root-relative array of the same shape.
    """
    pelvis = coords_mm[..., PELVIS_IDX : PELVIS_IDX + 1, :]
    return coords_mm - pelvis


def write_per_sequence_mats(
    predictions_mm: Dict[str, np.ndarray], out_dir: Path
) -> List[Path]:
    """Write ``TS{i}.mat`` files containing ``joint_3d_image`` variables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
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
    predictions: Dict[str, np.ndarray], test_root: Path, out_dir: Path
) -> Path:
    """Produce ``mpii_3dhp_prediction.mat`` for local evaluation."""
    import importlib.util

    out_dir.mkdir(parents=True, exist_ok=True)
    temp_npz = out_dir / ".tmp_predictions_for_local_eval.npz"
    np.savez_compressed(temp_npz, **predictions)

    local_mat = out_dir / "mpii_3dhp_prediction.mat"
    # Import on demand to keep script startup lightweight.
    spec = importlib.util.spec_from_file_location(
        "package_mpiinf3dhp_server_submission",
        REPO_ROOT / "scripts" / "package_mpiinf3dhp_server_submission.py",
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    module.package_submission(str(temp_npz), str(test_root), str(local_mat))

    temp_npz.unlink(missing_ok=True)
    return local_mat


def write_manifest(
    out_dir: Path,
    predictions: Dict[str, np.ndarray],
    method_name: str,
    config_path: Path,
    checkpoint_path: Path,
    test_set_dir: Path,
    test_root: Path | None,
) -> Path:
    manifest = {
        "method": method_name,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
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
        "  1. Visit https://mpi-inf-3dhp.is.tuebingen.mpg.de/\n"
        "  2. Log in / register with an academic email.\n"
        "  3. Upload submission.zip (TS1.mat .. TS6.mat).\n"
        "  4. Wait for the result email (usually within minutes to hours).\n\n"
        "Do not commit this zip to version control; it contains submission outputs.\n"
    )
    readme.write_text(text, encoding="utf-8")
    return readme


def zip_submission(out_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in out_dir.iterdir():
            if path.is_file() and path.suffix == ".mat":
                zf.write(path, arcname=path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_test_files(test_set_dir: Path) -> List[Path]:
    files: List[Path] = []
    for i in range(1, 7):
        path = test_set_dir / f"TS{i}_v14_multiview.npz"
        if not path.exists():
            raise FileNotFoundError(f"Expected test file not found: {path}")
        files.append(path)
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MPI-INF-3DHP official test-server submission from a v5 checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML training config (e.g. configs/ablations/v25_true_gt_v2_medium_a800.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained checkpoint .pth file.",
    )
    parser.add_argument(
        "--test_set_dir",
        type=Path,
        default=Path("data/webbridge/mpi_inf_3dhp/test_set"),
        help="Directory containing TS{i}_v14_multiview.npz files (default: data/webbridge/mpi_inf_3dhp/test_set).",
    )
    parser.add_argument(
        "--test_root",
        type=Path,
        default=None,
        help="Path to the downloaded MPI-INF-3DHP test set root (contains TS1..TS6 annot_data.mat). Optional.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Default: outputs/mpi_submissions/{model_name}/ where {model_name} is the config's name.",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default=None,
        help="Method name for the submission manifest. Default: taken from the YAML config's 'name' field.",
    )
    parser.add_argument(
        "--clip_len",
        type=int,
        default=13,
        help="Temporal clip length for inference (default: 13).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference batch size (default: 8).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride between consecutive clips (default: 1, sliding window).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device: 'auto' (default), 'cpu', or 'cuda:0' etc.",
    )
    parser.add_argument(
        "--no_zip",
        action="store_true",
        help="Do not create submission.zip; only write the .mat files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Build model architecture args from YAML config.
    model_args = namespace_from_yaml(args.config)

    # Discover test files and infer geometry.
    test_files = discover_test_files(args.test_set_dir)
    sample = np.load(test_files[0])
    n_views = int(sample["camera_K"].shape[0])
    n_joints = int(sample["points_2d"].shape[2])
    print(f"Test set: {len(test_files)} sequences, {n_views} views, {n_joints} joints")
    if n_joints != 17:
        print(
            f"Warning: test set has {n_joints} joints; MPI-INF-3DHP submissions "
            "usually expect 17 joints in the relevant skeleton."
        )

    # Build model.
    print("Building model from config...")
    model = build_model_from_args(model_args, n_joints=n_joints, n_views=n_views, device=device)
    model = model.to(device)

    # Load checkpoint.
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, str(args.checkpoint))
    model.eval()

    # Determine output directory and method name.
    with open(args.config, "r", encoding="utf-8") as fh:
        yaml_cfg = yaml.safe_load(fh)
    method_name = args.method_name or yaml_cfg.get("name") or args.config.stem
    output_dir = args.output_dir or (REPO_ROOT / "outputs" / "mpi_submissions" / method_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference on each test sequence.
    predictions_m: Dict[str, np.ndarray] = {}
    for test_file in test_files:
        print(f"Running inference on {test_file.name}...")
        seq_pred = infer_sequence(
            model,
            str(test_file),
            args.clip_len,
            args.batch_size,
            args.stride,
            device,
        )
        key = test_file.stem  # e.g. "TS1_v14_multiview"
        predictions_m[key] = seq_pred
        print(f"  -> {seq_pred.shape[0]} frames, {seq_pred.shape[1]} joints")

    # Convert to millimetres for the submission format.
    predictions_mm = {k: v * 1000.0 for k, v in predictions_m.items()}

    # Write per-sequence .mat files.
    write_per_sequence_mats(predictions_mm, output_dir)

    # Optionally write local-evaluation .mat.
    if args.test_root:
        print("Packaging local-evaluation mpii_3dhp_prediction.mat...")
        write_local_eval_mat(predictions_m, args.test_root, output_dir)

    # Write manifest and README.
    write_manifest(
        output_dir,
        predictions_mm,
        method_name,
        args.config,
        args.checkpoint,
        args.test_set_dir,
        args.test_root,
    )
    write_readme(output_dir)

    # Zip the submission.
    if not args.no_zip:
        zip_path = output_dir / "submission.zip"
        if zip_path.exists():
            zip_path.unlink()
        zip_submission(output_dir, zip_path)
        print(f"Saved submission zip -> {zip_path}")

    # Clean up the per-sequence .mat files are kept; only the zip is an extra file.
    print("\n" + "=" * 70)
    print("MPI-INF-3DHP submission package prepared")
    print("=" * 70)
    print(f"Method:     {method_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config:     {args.config}")
    print(f"Output dir: {output_dir.resolve()}")
    print("Per-sequence .mat files:")
    for i, key in enumerate(SEQUENCE_KEYS, start=1):
        print(f"  TS{i}.mat    <- {key} ({predictions_mm[key].shape[0]} frames)")
    if not args.no_zip:
        print(f"\nUpload {output_dir / 'submission.zip'} to the MPI-INF-3DHP server.")
    print("No data was uploaded automatically.")


if __name__ == "__main__":
    main()
