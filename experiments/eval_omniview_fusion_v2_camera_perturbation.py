"""Camera perturbation robustness evaluation for OmniMultiViewFusionV2.

Loads a trained ``OmniMultiViewFusionV2`` checkpoint and evaluates MPJPE on the
MPI-INF-3DHP test set while adding controlled noise to camera intrinsics and/or
extrinsics.  Results are written to JSON.

Usage
-----
    # Full evaluation (requires checkpoint + dataset)
    python experiments/eval_omniview_fusion_v2_camera_perturbation.py \
        --checkpoint outputs/omniview_fusion_v2_d128_no_graph.pth \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz

    # CPU/GPU smoke test on synthetic data
    python experiments/eval_omniview_fusion_v2_camera_perturbation.py --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import compute_all_metrics
from motionflow_mv.fusion.omniview_fusion_v2 import OmniMultiViewFusionV2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class TemporalClipDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str, clip_len: int, stride: int = 1):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()
        self.clip_len = clip_len
        self.stride = stride
        self.total_frames = self.points_2d.shape[0]
        self.num_clips = max(1, (self.total_frames - self.clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


def collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


# ---------------------------------------------------------------------------
# Synthetic smoke dataset
# ---------------------------------------------------------------------------

def _make_synthetic_cameras(n_views: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        self.joints_3d = joints_3d
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
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


# ---------------------------------------------------------------------------
# Model construction / loading
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
        n_joint_layers=args.n_joint_layers,
        return_pp_delta=False,
        return_covariance=True,
    )
    if j != 17:
        model.rebuild_graph(j, dataset="mpiinf3dhp")
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Checkpoint load: missing keys {missing[:10]}")
    if unexpected:
        print(f"Checkpoint load: unexpected keys ignored {unexpected[:10]}")


def infer_checkpoint_args(checkpoint_path: str) -> Dict[str, Any]:
    """Infer minimal architecture hyper-parameters from a saved state dict."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    view_pos_embed = state["view_pos_embed"]
    n_views, d = view_pos_embed.shape
    residual_hidden = state.get("residual_mlp.0.weight", torch.empty(128, 0)).shape[0]

    n_st_layers = 0
    while f"st_transformer.{n_st_layers}.self_attn.in_proj_weight" in state:
        n_st_layers += 1

    graph_num_layers = 1 if any(k.startswith("graph_joint_attention.") for k in state) else 0

    return {
        "d": d,
        "n_views": n_views,
        "residual_hidden": residual_hidden,
        "n_st_layers": max(1, n_st_layers),
        "graph_num_layers": graph_num_layers,
    }


# ---------------------------------------------------------------------------
# Camera perturbation helpers
# ---------------------------------------------------------------------------

def so3_exp(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = axis_angle.norm(dim=-1, keepdim=True)[..., None]
    k = axis_angle / (axis_angle.norm(dim=-1, keepdim=True) + 1e-8)
    K = torch.zeros(*axis_angle.shape[:-1], 3, 3, dtype=axis_angle.dtype, device=axis_angle.device)
    K[..., 0, 1] = -k[..., 2]
    K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]
    K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]
    K[..., 2, 1] = k[..., 0]
    I = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    I = I.view(*([1] * (axis_angle.ndim - 1)), 3, 3).expand(*axis_angle.shape[:-1], 3, 3)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def corrupt_intrinsics(K: torch.Tensor, focal_err: float, cxcy_err: float) -> torch.Tensor:
    K_out = K.clone()
    K_out[..., 0, 0] *= 1 + focal_err
    K_out[..., 1, 1] *= 1 + focal_err
    K_out[..., 0, 2] += cxcy_err
    K_out[..., 1, 2] += cxcy_err
    return K_out


def corrupt_extrinsics(
    R: torch.Tensor,
    t: torch.Tensor,
    rot_std_deg: float,
    trans_std: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    R_out = R.clone()
    t_out = t.clone()
    if rot_std_deg > 0:
        noise = torch.randn(R.shape[0], 3, device=R.device, dtype=R.dtype) * np.deg2rad(rot_std_deg)
        delta_R = so3_exp(noise)
        R_out = torch.einsum("vij,vjk->vik", delta_R, R)
    if trans_std > 0:
        t_out = t + torch.randn_like(t) * trans_std
    return R_out, t_out


class PerturbedDataset(torch.utils.data.Dataset):
    """Wrap a base dataset and apply per-sample camera perturbations."""

    def __init__(self, base: torch.utils.data.Dataset, cfg: Dict[str, float], seed: int):
        self.base = base
        self.cfg = cfg
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y, K, R, t = self.base[idx]
        rng = np.random.default_rng(self.seed + idx)
        if self.cfg.get("focal_err", 0.0) != 0.0 or self.cfg.get("cxcy_err", 0.0) != 0.0:
            K = corrupt_intrinsics(K, self.cfg.get("focal_err", 0.0), self.cfg.get("cxcy_err", 0.0))
        if self.cfg.get("rot_std", 0.0) != 0.0 or self.cfg.get("trans_std", 0.0) != 0.0:
            rot_std = self.cfg.get("rot_std", 0.0)
            trans_std = self.cfg.get("trans_std", 0.0)
            # Deterministic per-sample noise seeded by the global seed + index.
            if rot_std > 0:
                noise = torch.from_numpy(rng.normal(0.0, np.deg2rad(rot_std), size=(R.shape[0], 3))).float()
                delta_R = so3_exp(noise)
                R = torch.einsum("vij,vjk->vik", delta_R, R)
            if trans_std > 0:
                t = t + torch.from_numpy(rng.normal(0.0, trans_std, size=t.shape)).float()
        return x, y, K, R, t


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            out = model(xb, K=K, R=R, t=t)
            pred = out[0]
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], 3) * 1000.0
    gts = np.concatenate(gts, axis=0).reshape(-1, gts[0].shape[-2], 3) * 1000.0
    return compute_all_metrics(preds, gts)


def evaluate_perturbation_conditions(
    model: torch.nn.Module,
    base_dataset: torch.utils.data.Dataset,
    conditions: Dict[str, Dict[str, float]],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for name, cfg in conditions.items():
        print(f"  -> {name}: {cfg}")
        perturbed = PerturbedDataset(base_dataset, cfg, seed=seed)
        loader = torch.utils.data.DataLoader(
            perturbed,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        report = _evaluate_loader(model, loader, device)
        results[name] = {
            "mpjpe_mm": float(report["mpjpe"]),
            "pa_mpjpe_mm": float(report["pa_mpjpe"]),
            "config": cfg,
        }
        print(
            f"     MPJPE={report['mpjpe']:.2f}mm "
            f"PA-MPJPE={report['pa_mpjpe']:.2f}mm"
        )
    return results


def perturbation_conditions(smoke: bool = False) -> Dict[str, Dict[str, float]]:
    conditions = {
        "clean": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 0.0},
        "rot_0.5_deg": {"rot_std": 0.5, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 0.0},
        "rot_1.0_deg": {"rot_std": 1.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 0.0},
        "rot_2.0_deg": {"rot_std": 2.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 0.0},
        "trans_5mm": {"rot_std": 0.0, "trans_std": 0.005, "focal_err": 0.0, "cxcy_err": 0.0},
        "trans_10mm": {"rot_std": 0.0, "trans_std": 0.010, "focal_err": 0.0, "cxcy_err": 0.0},
        "trans_20mm": {"rot_std": 0.0, "trans_std": 0.020, "focal_err": 0.0, "cxcy_err": 0.0},
        "focal_1pct": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.01, "cxcy_err": 0.0},
        "focal_2pct": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.02, "cxcy_err": 0.0},
        "focal_5pct": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.05, "cxcy_err": 0.0},
        "principal_3px": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 3.0},
        "principal_5px": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 5.0},
        "principal_10px": {"rot_std": 0.0, "trans_std": 0.0, "focal_err": 0.0, "cxcy_err": 10.0},
    }
    if smoke:
        conditions = {
            "clean": conditions["clean"],
            "rot_0.5_deg": conditions["rot_0.5_deg"],
            "focal_1pct": conditions["focal_1pct"],
            "principal_3px": conditions["principal_3px"],
        }
    return conditions


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def write_json(out_path: Path, results: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Camera-perturbation robustness for OmniMultiViewFusionV2",
    )
    # Inputs
    parser.add_argument("--checkpoint", type=str, default="outputs/omniview_fusion_v2_d128_no_graph.pth",
                        help="Path to trained OmniMultiViewFusionV2 checkpoint")
    parser.add_argument("--dataset", type=str, default="data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz",
                        help="Path to MPI-INF-3DHP .npz (S2/Seq1)")
    parser.add_argument("--smoke", action="store_true", help="CPU/GPU smoke test on synthetic data")
    # Model
    parser.add_argument("--d", type=int, default=128, help="Model feature dimension")
    parser.add_argument("--residual_hidden", type=int, default=128, help="Residual MLP hidden size")
    parser.add_argument("--n_st_layers", type=int, default=2, help="Spatio-temporal transformer layers")
    parser.add_argument("--graph_num_layers", type=int, default=1, help="Graph-joint attention layers")
    parser.add_argument("--n_joint_layers", type=int, default=0, help="Dense joint-level transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    # Evaluation
    parser.add_argument("--clip_len", type=int, default=13, help="Temporal clip length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--val_stride", type=int, default=1, help="Stride for validation clips")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Outputs
    parser.add_argument("--out_json", type=str, default="outputs/eval_omniview_fusion_v2_camera_perturbation.json",
                        help="JSON output path")
    args = parser.parse_args()

    if args.smoke:
        args.clip_len = 9
        args.batch_size = 2

    if Path(args.checkpoint).exists():
        inferred = infer_checkpoint_args(args.checkpoint)
        args.d = inferred["d"]
        args.residual_hidden = inferred["residual_hidden"]
        args.n_st_layers = inferred["n_st_layers"]
        args.graph_num_layers = inferred["graph_num_layers"]
        if args.smoke:
            args.n_views = inferred["n_views"]
    elif args.smoke:
        args.checkpoint = "__smoke__"
    else:
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    if not args.smoke and not Path(args.dataset).exists():
        parser.error(f"Dataset not found: {args.dataset}")

    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    if args.smoke:
        print("Smoke mode: using synthetic dataset")
        n_views = getattr(args, "n_views", 4) if getattr(args, "n_views", None) is not None else 4
        K, R, t = _make_synthetic_cameras(n_views=n_views)
        n_joints = 17
        dataset = SyntheticSmokeDataset(K, R, t, n_frames=60, n_joints=n_joints, clip_len=args.clip_len)
    else:
        data = np.load(args.dataset)
        n_views = int(data["camera_K"].shape[0])
        n_joints = int(data["points_2d"].shape[2])
        dataset = TemporalClipDataset(args.dataset, args.clip_len, stride=args.val_stride)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Model
    model = build_model(args, n_views=n_views, j=n_joints).to(device)
    if args.checkpoint != "__smoke__":
        load_checkpoint(model, args.checkpoint)
    else:
        print("No checkpoint provided; using freshly initialised model for smoke test")
    model.eval()

    # Clean baseline
    print("Clean baseline...")
    baseline_report = _evaluate_loader(model, loader, device)
    print(
        f"Clean: MPJPE={baseline_report['mpjpe']:.2f}mm "
        f"PA-MPJPE={baseline_report['pa_mpjpe']:.2f}mm"
    )

    # Perturbation matrix
    conditions = perturbation_conditions(smoke=args.smoke)
    print("Camera perturbation matrix...")
    perturbation_results = evaluate_perturbation_conditions(
        model,
        dataset,
        conditions,
        args.batch_size,
        device,
        seed=args.seed,
    )

    results = {
        "args": vars(args),
        "baseline": {k: float(v) for k, v in baseline_report.items()
                     if not ("_per_joint" in k or k.startswith("per_joint_"))},
        "perturbations": perturbation_results,
    }

    out_json = Path(args.out_json)
    write_json(out_json, results)
    print(f"Saved JSON -> {out_json}")


if __name__ == "__main__":
    main()
