"""Test-set inference for OmniMultiViewFusionV2 on MPI-INF-3DHP TS1-TS6.

Loads a trained OmniMultiViewFusionV2 checkpoint and produces 3-D pose
predictions for every frame of the MPI-INF-3DHP test set.  Predictions are
written to a single ``.npz`` file with one array per test sequence.  The
script supports a CPU/GPU smoke mode that runs on synthetic data without
requiring the real checkpoint or test set.

Usage
-----
    # Full inference on the standard test set.
    python experiments/infer_mpiinf3dhp_test_set_omniview_v2.py \
        --checkpoint outputs/omniview_fusion_v2_mpiinf3dhp.pth \
        --out_npz outputs/omniview_fusion_v2_test_set_predictions.npz

    # Smoke test (synthetic data, fresh model).
    python experiments/infer_mpiinf3dhp_test_set_omniview_v2.py --smoke
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.omniview_fusion_v2 import OmniMultiViewFusionV2
from motionflow_mv.training.trainer_v2 import checkpoint_eval_state_dict


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class TemporalClipDataset(torch.utils.data.Dataset):
    """Yield overlapping temporal clips from a canonical multi-view .npz."""

    def __init__(self, npz_path: str, clip_len: int, stride: int = 1):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()  # (T, V, J, 2)
        self.confidences = torch.from_numpy(data["confidences"]).float()  # (T, V, J)
        self.K = torch.from_numpy(data["camera_K"]).float()  # (V, 3, 3)
        self.R = torch.from_numpy(data["camera_R"]).float()  # (V, 3, 3)
        self.t = torch.from_numpy(data["camera_t"]).float()  # (V, 3)

        self.clip_len = clip_len
        self.stride = stride
        self.total_frames = self.points_2d.shape[0]
        self.num_clips = max(1, (self.total_frames - clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )  # (T, V, J, 3)
        return x, self.K, self.R, self.t, start


def collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    x = torch.stack([b[0] for b in batch], dim=0)
    K = torch.stack([b[1] for b in batch], dim=0)
    R = torch.stack([b[2] for b in batch], dim=0)
    t = torch.stack([b[3] for b in batch], dim=0)
    starts = np.array([b[4] for b in batch], dtype=np.int64)
    return x, K, R, t, starts


# ---------------------------------------------------------------------------
# Synthetic smoke dataset
# ---------------------------------------------------------------------------

def _make_synthetic_cameras(n_views: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a circular rig of pinhole cameras and return K, R, t tensors."""
    K_list, R_list, t_list = [], [], []
    for i in range(n_views):
        theta = 2 * math.pi * i / n_views
        c = np.array([3.0 * np.cos(theta), 3.0 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        K_list.append(K)
        R_list.append(R)
        t_list.append(t)
    return (
        torch.from_numpy(np.stack(K_list, axis=0)).float(),
        torch.from_numpy(np.stack(R_list, axis=0)).float(),
        torch.from_numpy(np.stack(t_list, axis=0)).float(),
    )


class SyntheticSmokeDataset(torch.utils.data.Dataset):
    """Tiny synthetic multi-view pose dataset for smoke tests."""

    def __init__(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        n_frames: int = 60,
        n_joints: int = 17,
        clip_len: int = 9,
        noise_std: float = 0.5,
    ):
        self.K = K
        self.R = R
        self.t = t
        self.n_joints = n_joints
        self.clip_len = clip_len
        self.noise_std = noise_std

        torch.manual_seed(42)
        joints_3d = torch.randn(n_frames, n_joints, 3) * 0.3
        for _ in range(2):
            joints_3d[1:-1] = 0.5 * joints_3d[1:-1] + 0.25 * (joints_3d[:-2] + joints_3d[2:])

        points_2d = self._project_points(joints_3d, K, R, t)
        if noise_std > 0:
            points_2d = points_2d + torch.randn_like(points_2d) * noise_std

        self.points_2d = points_2d
        self.confidences = torch.ones_like(points_2d[..., 0])
        self.total_frames = n_frames
        self.num_clips = max(1, self.total_frames - self.clip_len + 1)

    @staticmethod
    def _project_points(
        joints_3d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        X = joints_3d
        t = t[:, None, None, :]
        X_cam = torch.einsum("vab,fjb->vfja", R, X) + t
        z = X_cam[..., 2:3].clamp(min=1e-6)
        uv = torch.matmul(K[:, None, None], (X_cam / z)[..., None])
        points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
        return points_2d.permute(1, 0, 2, 3)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx: int):
        start = idx
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        return x, self.K, self.R, self.t, start


# ---------------------------------------------------------------------------
# Model construction / checkpoint loading
# ---------------------------------------------------------------------------

def build_model(args: argparse.Namespace, n_views: int, j: int) -> OmniMultiViewFusionV2:
    model = OmniMultiViewFusionV2(
        j=j,
        d=args.d,
        n_views=n_views,
        n_heads=args.n_heads,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        graph_num_layers=args.graph_num_layers,
        return_pp_delta=False,
        return_covariance=True,
    )
    if j != 17:
        model.rebuild_graph(j, dataset="mpiinf3dhp")
    return model


def _infer_checkpoint_joints(state: Dict[str, Any], n_views: int) -> int | None:
    """Try to infer the joint count the checkpoint was trained with."""
    # The graph edge index is the most reliable indicator of the skeleton size.
    edge_key = "graph_joint_attention.edge_index"
    if edge_key in state:
        # shape (2, E); E depends on (V, J) and edge types.
        # We can solve for J by comparing against known counts.
        E = state[edge_key].shape[1]
        # edge count for H36M 17 joints vs MPI-INF-3DHP 28 joints with 14 views.
        # build_edge_index from motionflow_mv.fusion.graph_joint_relation can be
        # used, but here we just compare with the canonical view counts.
        for candidate in (17, 28):
            try:
                from motionflow_mv.fusion.graph_joint_relation import build_edge_index
                from motionflow_mv.fusion.prototypes.cross_view_graph_attention import (
                    H36M_17_PARENTS,
                    H36M_17_SYMMETRY_PAIRS,
                    MPI_INF_3DHP_28_PARENTS,
                    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
                )
                if candidate == 17:
                    parents, symmetry = H36M_17_PARENTS, H36M_17_SYMMETRY_PAIRS
                else:
                    parents, symmetry = MPI_INF_3DHP_28_PARENTS, MPI_INF_3DHP_28_SYMMETRY_PAIRS
                edge_index, edge_type = build_edge_index(parents, symmetry, n_views, candidate)
                if edge_index.shape[1] == E:
                    return candidate
            except Exception:
                continue
    return None


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, n_views: int, test_joints: int) -> int | None:
    """Load a checkpoint, tolerating skeleton/joint-count mismatches.

    Returns the joint count the checkpoint was trained with, or None if it
    could not be inferred.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint_eval_state_dict(state)

    checkpoint_joints = _infer_checkpoint_joints(state, n_views)

    # Filter the state dict to avoid shape mismatches (e.g. graph edge buffers).
    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for k, v in state.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)
        # unexpected keys are silently ignored; strict=False will report them.

    if skipped:
        print(
            f"Warning: checkpoint joint count likely differs from test-set "
            f"joint count ({checkpoint_joints} vs {test_joints}). Skipping "
            f"graph buffers with mismatched shapes: {skipped[:5]}"
        )

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"Checkpoint load: missing keys {missing[:10]}")
    if unexpected:
        print(f"Checkpoint load: unexpected keys ignored {unexpected[:10]}")

    return checkpoint_joints


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_sequence(
    model: torch.nn.Module,
    npz_path: str,
    clip_len: int,
    batch_size: int,
    stride: int,
    device: torch.device,
) -> np.ndarray:
    """Run inference on a single test sequence and return per-frame 3D poses."""
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
            pred = out[0].cpu()  # (B, T, J, 3)

            B, T, J, _ = pred.shape
            for i in range(B):
                start = starts[i]
                end = start + T
                accum[start:end] += pred[i]
                counts[start:end] += 1.0

    counts = counts.clamp(min=1.0)
    per_frame = accum / counts
    return per_frame.numpy()


def run_smoke_inference(args: argparse.Namespace, device: torch.device) -> Dict[str, np.ndarray]:
    """Run inference on a synthetic smoke dataset and return predictions."""
    K, R, t = _make_synthetic_cameras(n_views=4)
    dataset = SyntheticSmokeDataset(K, R, t, n_frames=60, n_joints=17, clip_len=args.clip_len)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = build_model(args, n_views=4, j=17).to(device)
    total_frames = dataset.total_frames
    n_joints = 17
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
                start = starts[i]
                end = start + T
                accum[start:end] += pred[i]
                counts[start:end] += 1.0

    counts = counts.clamp(min=1.0)
    per_frame = accum / counts
    return {"smoke": per_frame.numpy()}


def discover_test_files(test_set_dir: Path) -> List[Path]:
    """Return TS1..TS6 .npz files in the test set directory."""
    files: List[Path] = []
    for i in range(1, 7):
        path = test_set_dir / f"TS{i}_v14_multiview.npz"
        if path.exists():
            files.append(path)
        else:
            print(f"Warning: expected test file not found: {path}")
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OmniMultiViewFusionV2 inference on MPI-INF-3DHP test set",
    )
    # Inputs
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained OmniMultiViewFusionV2 checkpoint")
    parser.add_argument("--test_set_dir", type=str, default="data/webbridge/mpi_inf_3dhp/test_set", help="Directory containing TS{i}_v14_multiview.npz")
    parser.add_argument("--smoke", action="store_true", help="CPU/GPU smoke test on synthetic data")
    # Model
    parser.add_argument("--d", type=int, default=128, help="Model feature dimension")
    parser.add_argument("--residual_hidden", type=int, default=128, help="Residual MLP hidden size")
    parser.add_argument("--n_st_layers", type=int, default=2, help="Spatio-temporal transformer layers")
    parser.add_argument("--graph_num_layers", type=int, default=1, help="Graph-joint attention layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    # Inference
    parser.add_argument("--clip_len", type=int, default=13, help="Temporal clip length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--stride", type=int, default=1, help="Stride between consecutive clips (1 = sliding window)")
    parser.add_argument("--device", type=str, default="auto", help="Device string ('auto' selects cuda if available)")
    # Outputs
    parser.add_argument("--out_npz", type=str, default="outputs/omniview_fusion_v2_test_set_predictions.npz", help="Output .npz path")
    args = parser.parse_args()

    if args.smoke:
        args.clip_len = 9
        args.batch_size = 2
        if args.checkpoint is None:
            args.checkpoint = "__smoke__"
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required unless --smoke is set")

    return args


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    if args.smoke:
        print("Smoke mode: using synthetic dataset")
        predictions = run_smoke_inference(args, device)
    else:
        test_files = discover_test_files(Path(args.test_set_dir))
        if not test_files:
            raise FileNotFoundError(f"No test files found in {args.test_set_dir}")

        # Use the first file to determine skeleton/camera properties.
        sample = np.load(test_files[0])
        n_views = int(sample["camera_K"].shape[0])
        test_joints = int(sample["points_2d"].shape[2])

        print(f"Test set: {len(test_files)} sequences, {n_views} views, {test_joints} joints")
        if test_joints != 17:
            print(f"Warning: test set has {test_joints} joints; MPI-INF-3DHP submissions usually expect 17 joints.")

        # Build model for the test-set skeleton.
        model = build_model(args, n_views=n_views, j=test_joints).to(device)

        # Load checkpoint, tolerating joint-count mismatches.
        checkpoint_joints = load_checkpoint(model, args.checkpoint, n_views, test_joints)
        if checkpoint_joints is not None and checkpoint_joints != test_joints:
            print(
                f"Warning: checkpoint was trained with {checkpoint_joints} joints "
                f"but test set has {test_joints} joints. Graph has been rebuilt for the test set."
            )

        model.eval()

        predictions: Dict[str, np.ndarray] = {}
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
            key = test_file.stem
            predictions[key] = seq_pred
            print(f"  -> {seq_pred.shape[0]} frames, {seq_pred.shape[1]} joints")

    # Save predictions.
    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **predictions)
    print(f"Saved predictions -> {out_path}")


if __name__ == "__main__":
    main()
