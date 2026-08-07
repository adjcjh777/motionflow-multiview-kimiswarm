"""Train OmniMultiViewFusionV3 on MPI-INF-3DHP.

This script is a self-contained trainer for ``OmniMultiViewFusionV3`` that uses
``motionflow_mv.training.trainer_v2.TrainerV2`` (cosine LR, warmup, gradient
clipping, AMP, EMA).  It follows the same data-loading pattern as
``experiments/train_omniview_fusion_v2_mpiinf3dhp.py`` and adds the Omni-specific
losses: 3D MPJPE, visibility BCE, uncertainty NLL, temporal consistency and bone-
length regularisation.

Usage
-----
    # CPU / GPU smoke (1 epoch, synthetic data)
    python experiments/train_omniview_fusion_v3_mpiinf3dhp.py --smoke

    # Full MPI-INF-3DHP run
    python experiments/train_omniview_fusion_v3_mpiinf3dhp.py \
        --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
                data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
                data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
                data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --d 128 --residual_hidden 128 --epochs 30 \
        --use_multiscale_fusion --use_camera_conditioning --use_epipolar_bias
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.omniview_fusion_v3 import OmniMultiViewFusionV3
from motionflow_mv.fusion.prototypes.cross_view_graph_attention import (
    H36M_17_PARENTS,
    MPI_INF_3DHP_28_PARENTS,
)
from motionflow_mv.training.trainer_v2 import TrainerV2, build_lr_scheduler, checkpoint_eval_state_dict


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parent_indices(j: int) -> List[int]:
    """Return the parent list for the active skeleton."""
    if j == 17:
        return H36M_17_PARENTS
    if j == 28:
        return MPI_INF_3DHP_28_PARENTS
    raise ValueError(f"No built-in parent list for j={j}")


# ---------------------------------------------------------------------------
# Data loading (same pattern as train_omniview_fusion_v2_mpiinf3dhp.py)
# ---------------------------------------------------------------------------

class TemporalClipDataset(torch.utils.data.Dataset):
    """Yield clips (T, V, J, 3) from a long canonical .npz sequence."""

    def __init__(self, npz_path: str, clip_len: int, stride: int = 1):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()  # (T, V, J, 2)
        self.confidences = torch.from_numpy(data["confidences"]).float()  # (T, V, J)
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()  # (T, J, 3)
        self.K = torch.from_numpy(data["camera_K"]).float()  # (V, 3, 3)
        self.R = torch.from_numpy(data["camera_R"]).float()  # (V, 3, 3)
        self.t = torch.from_numpy(data["camera_t"]).float()  # (V, 3)

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
        )  # (T, V, J, 3)
        y = self.joints_3d[start:end]  # (T, J, 3)
        return x, y, self.K, self.R, self.t


class RandomClipDataset(torch.utils.data.Dataset):
    """Sample random clips from a sequence; useful for train set augmentation."""

    def __init__(self, npz_path: str, clip_len: int, n_samples: int = 2000):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()
        self.clip_len = clip_len
        self.n_samples = n_samples
        self.total_frames = self.points_2d.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        start = random.randint(0, max(0, self.total_frames - self.clip_len))
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
# Synthetic smoke dataset (CPU-friendly, no external data required)
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
    K = torch.from_numpy(np.stack(K_list, axis=0)).float()
    R = torch.from_numpy(np.stack(R_list, axis=0)).float()
    t = torch.from_numpy(np.stack(t_list, axis=0)).float()
    return K, R, t


class SyntheticSmokeDataset(torch.utils.data.Dataset):
    """Tiny synthetic multi-view pose dataset for smoke tests."""

    def __init__(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        n_frames: int = 100,
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
        """Project world points into all views."""
        X = joints_3d  # (F, J, 3)
        t = t[:, None, None, :]
        X_cam = torch.einsum("vab,fjb->vfja", R, X) + t  # (V, F, J, 3)
        z = X_cam[..., 2:3].clamp(min=1e-6)
        uv = torch.matmul(K[:, None, None], (X_cam / z)[..., None])  # (V, F, J, 3, 1)
        points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
        return points_2d.permute(1, 0, 2, 3)  # (F, V, J, 2)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx: int):
        start = idx
        end = start + self.clip_len
        x = torch.cat([self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)], dim=-1)
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def augment_clip(
    x: torch.Tensor,
    *,
    noise_std: float = 0.5,
    dropout_rate: float = 0.0,
    view_dropout_rate: float = 0.0,
    min_views: int = 2,
) -> torch.Tensor:
    """Lightweight per-clip augmentation with optional view dropout."""
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = (torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device) > dropout_rate).float()
        x[..., 2] = x[..., 2] * mask
    if view_dropout_rate > 0:
        B = x.shape[0]
        V = x.shape[2]
        view_mask = (torch.rand(B, V, device=x.device) > view_dropout_rate).float()
        for i in range(B):
            kept = view_mask[i].nonzero(as_tuple=True)[0]
            if kept.numel() < min_views:
                dropped = (view_mask[i] == 0).nonzero(as_tuple=True)[0]
                needed = min_views - kept.numel()
                if needed > 0 and dropped.numel() > 0:
                    perm = torch.randperm(dropped.numel())
                    extra = dropped[perm[:needed]]
                    view_mask[i, extra] = 1.0
        x[..., 2] = x[..., 2] * view_mask.view(B, 1, V, 1)
    return x


# ---------------------------------------------------------------------------
# Auxiliary losses
# ---------------------------------------------------------------------------

def project_points_3d_to_2d(
    points_3d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Project 3-D points into each view.

    Args:
        points_3d: (B, T, J, 3)
        K, R, t: (B, V, 3, 3) / (B, V, 3)

    Returns:
        uv: (B, T, V, J, 2)
    """
    B, T, J, _ = points_3d.shape
    V = K.shape[1]
    K = K.unsqueeze(1).expand(-1, T, -1, -1, -1)
    R = R.unsqueeze(1).expand(-1, T, -1, -1, -1)
    t = t.unsqueeze(1).expand(-1, T, -1, -1)
    X_cam = torch.einsum("btvac,btjc->btvja", R, points_3d) + t.unsqueeze(3)
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv_h = torch.einsum("btvik,btvjk->btvji", K, X_cam / z)
    return uv_h[..., :2]


def uncertainty_nll_loss(
    pred_3d: torch.Tensor,
    points_2d: torch.Tensor,
    confidences: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    L: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood of 2-D reprojection residuals under L L^T."""
    uv_pred = project_points_3d_to_2d(pred_3d, K, R, t)  # (B, T, V, J, 2)
    r = uv_pred - points_2d  # (B, T, V, J, 2)
    valid = confidences > 0  # (B, T, V, J)

    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_3d.device, dtype=pred_3d.dtype)

    r_flat = r.reshape(-1, 2)[valid.reshape(-1)]
    L_flat = L.reshape(-1, 2, 2)[valid.reshape(-1)]

    y = torch.linalg.solve(L_flat, r_flat.unsqueeze(-1)).squeeze(-1)
    a = torch.linalg.solve(L_flat.transpose(-2, -1), y.unsqueeze(-1)).squeeze(-1)
    mahalanobis = (r_flat * a).sum(dim=-1)

    l00 = L_flat[..., 0, 0].clamp(min=1e-6)
    l11 = L_flat[..., 1, 1].clamp(min=1e-6)
    log_det = 2.0 * (torch.log(l00) + torch.log(l11))

    nll = 0.5 * (mahalanobis + log_det + 2.0 * math.log(2.0 * math.pi))
    return nll.mean()


def bone_length_loss(pred_3d: torch.Tensor, target_3d: torch.Tensor, parents: List[int]) -> torch.Tensor:
    """MSE between predicted and target bone vectors."""
    loss = torch.tensor(0.0, device=pred_3d.device, dtype=pred_3d.dtype)
    count = 0
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        pred_bone = pred_3d[..., child, :] - pred_3d[..., parent, :]
        tgt_bone = target_3d[..., child, :] - target_3d[..., parent, :]
        loss = loss + F.mse_loss(pred_bone, tgt_bone)
        count += 1
    return loss / max(1, count)


# ---------------------------------------------------------------------------
# Trainer wrapper
# ---------------------------------------------------------------------------

def build_compute_loss(args: Namespace):
    """Build the compute_loss closure used by TrainerV2."""
    parents = None
    if args.bone_loss_weight > 0.0:
        try:
            parents = get_parent_indices(args.j)
        except ValueError:
            parents = None

    def compute_loss(
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x, y, K, R, t = batch
        x = x.to(device)
        y = y.to(device)
        K = K.to(device)
        R = R.to(device)
        t = t.to(device)

        x = augment_clip(
            x.clone(),
            noise_std=args.noise_std if not args.smoke else 0.0,
            dropout_rate=args.confidence_dropout,
            view_dropout_rate=args.view_dropout_rate,
            min_views=args.min_views,
        )

        out = model(x, K=K, R=R, t=t)
        pred_3d = out[0]
        visibility = out[2]
        L = out[3]
        epi_loss = out[4]

        loss = F.mse_loss(pred_3d, y)
        metrics: Dict[str, Any] = {
            "mpjpe": (pred_3d - y).norm(dim=-1).mean().item(),
        }

        if epi_loss is not None:
            loss = loss + epi_loss

        if args.visibility_loss_weight > 0.0:
            visible_target = (x[..., 2] > 0).float()
            vis_loss = F.binary_cross_entropy(visibility, visible_target)
            loss = loss + args.visibility_loss_weight * vis_loss
            metrics["vis_loss"] = vis_loss.item()

        if args.uncertainty_loss_weight > 0.0:
            nll = uncertainty_nll_loss(pred_3d, x[..., :2], x[..., 2], K, R, t, L)
            loss = loss + args.uncertainty_loss_weight * nll
            metrics["nll"] = nll.item()

        if args.temporal_loss_weight > 0.0 and pred_3d.shape[1] > 1:
            vel_pred = pred_3d[:, 1:] - pred_3d[:, :-1]
            vel_tgt = y[:, 1:] - y[:, :-1]
            temp_loss = (vel_pred - vel_tgt).norm(dim=-1).mean()
            loss = loss + args.temporal_loss_weight * temp_loss
            metrics["temp_loss"] = temp_loss.item()

        if args.bone_loss_weight > 0.0 and parents is not None:
            bl = bone_length_loss(pred_3d, y, parents)
            loss = loss + args.bone_loss_weight * bl
            metrics["bone_loss"] = bl.item()

        return loss, metrics

    return compute_loss


def build_eval_metric():
    """Validation metric: MSE and MPJPE."""

    def eval_metric(
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> Dict[str, Any]:
        x, y, K, R, t = batch
        x = x.to(device)
        y = y.to(device)
        K = K.to(device)
        R = R.to(device)
        t = t.to(device)

        with torch.no_grad():
            out = model(x, K=K, R=R, t=t)
            pred_3d = out[0]
            loss = F.mse_loss(pred_3d, y)
            mpjpe = (pred_3d - y).norm(dim=-1).mean()
        return {"loss": loss, "mpjpe": mpjpe}

    return eval_metric


class OmniMultiViewTrainer(TrainerV2):
    """TrainerV2 pre-wired with the OmniMultiViewFusionV3 loss mix."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        args: Namespace,
        **kwargs: Any,
    ):
        compute_loss = build_compute_loss(args)
        super().__init__(
            model,
            optimizer,
            device,
            compute_loss=compute_loss,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Warm-start helpers
# ---------------------------------------------------------------------------

def load_warm_start(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Load a checkpoint, tolerating extra/missing keys for warm-starting."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint_eval_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warm-start: missing keys (expected for new heads): {missing[:10]}")
    if unexpected:
        print(f"Warm-start: unexpected keys ignored: {unexpected[:10]}")
    print(f"Warm-started from {checkpoint_path}")


def freeze_old_params(model: torch.nn.Module) -> None:
    """Freeze everything except the new v3-specific heads."""
    new_prefixes = (
        "graph_joint_attention",
        "visibility_head",
        "omni_joint_attn",
        "multiscale_fusion",
        "camera_conditioning",
        "epipolar_bias",
    )
    for name, param in model.named_parameters():
        if not name.startswith(new_prefixes):
            param.requires_grad = False


def unfreeze_all(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Train OmniMultiViewFusionV3 on MPI-INF-3DHP",
    )
    # Data
    parser.add_argument("--train", type=str, nargs="+", default=None, help="Train .npz files")
    parser.add_argument("--val", type=str, default=None, help="Validation .npz file")
    parser.add_argument("--smoke", action="store_true", help="1-epoch CPU/GPU smoke test on synthetic data")
    # Model
    parser.add_argument("--d", type=int, default=128, help="Model feature dimension")
    parser.add_argument("--residual_hidden", type=int, default=128, help="Residual MLP hidden size")
    parser.add_argument("--n_st_layers", type=int, default=2, help="Spatio-temporal transformer layers")
    parser.add_argument("--graph_num_layers", type=int, default=1, help="Graph-joint attention layers")
    parser.add_argument("--n_joint_layers", type=int, default=0, help="Dense joint-level transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--epipolar_loss_weight", type=float, default=0.05, help="Epipolar loss weight passed to the model")
    # New v3 components
    parser.add_argument("--use_multiscale_fusion", action="store_true", help="Enable hierarchical multi-scale fusion")
    parser.add_argument("--use_camera_conditioning", action="store_true", help="Enable camera conditioning")
    parser.add_argument("--use_epipolar_bias", action="store_true", help="Enable epipolar-biased ST attention")
    parser.add_argument("--multiscale_scales", type=int, nargs="+", default=(1, 2, 4), help="Multi-scale fusion downsample factors")
    parser.add_argument("--camera_condition_dim", type=int, default=32, help="Camera condition hidden dimension")
    parser.add_argument("--epipolar_temperature", type=float, default=10.0, help="Epipolar bias temperature")
    # Training
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--clip_len", type=int, default=13, help="Temporal clip length")
    parser.add_argument("--train_samples", type=int, default=4000, help="Random clips per train sequence")
    parser.add_argument("--val_stride", type=int, default=1, help="Validation clip stride")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr_cosine", action="store_true", help="Use cosine LR schedule with warmup")
    parser.add_argument("--lr_warmup_epochs", type=int, default=0, help="Warmup epochs (requires --lr_cosine)")
    parser.add_argument("--lr_min", type=float, default=0.0, help="Minimum LR for cosine schedule")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Global gradient clipping norm")
    parser.add_argument("--amp", action="store_true", help="Enable AMP (CUDA only)")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay (0 disables)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Augmentation
    parser.add_argument("--noise_std", type=float, default=0.5, help="2-D observation noise std (pixels)")
    parser.add_argument("--confidence_dropout", type=float, default=0.0, help="Confidence dropout rate")
    parser.add_argument("--view_dropout_rate", type=float, default=0.0, help="Probability of dropping a full view")
    parser.add_argument("--min_views", type=int, default=2, help="Minimum kept views during view dropout")
    # Loss weights
    parser.add_argument("--visibility_loss_weight", type=float, default=0.1, help="Visibility BCE weight")
    parser.add_argument("--uncertainty_loss_weight", type=float, default=0.05, help="Uncertainty NLL weight")
    parser.add_argument("--temporal_loss_weight", type=float, default=0.02, help="Temporal consistency weight")
    parser.add_argument("--bone_loss_weight", type=float, default=0.05, help="Bone-length consistency weight")
    # Warm-start
    parser.add_argument("--warm_start", type=str, default=None, help="Path to v2/v3 checkpoint for warm-starting")
    parser.add_argument("--warm_start_freeze_epochs", type=int, default=0, help="Freeze encoder/transformer for N epochs after warm-start")
    # I/O
    parser.add_argument("--output", type=str, default="outputs/omniview_fusion_v3_mpiinf3dhp.pth", help="Checkpoint path")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        args.d = 32
        args.residual_hidden = 64
        args.n_st_layers = 1
        args.graph_num_layers = 1
        args.batch_size = 2
        args.clip_len = 9
        args.train_samples = 16
        args.val_stride = 1
        args.train_samples = 16
        if args.train is None or args.val is None:
            args.use_synthetic = True
        else:
            args.use_synthetic = False
    else:
        args.use_synthetic = False
        if args.train is None or args.val is None:
            parser.error("--train and --val are required unless --smoke is set")

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.use_synthetic:
        print("Using synthetic smoke dataset")
        K, R, t = _make_synthetic_cameras(n_views=4)
        n_joints = 17
        train_dataset = SyntheticSmokeDataset(
            K, R, t, n_frames=50, n_joints=n_joints, clip_len=args.clip_len
        )
        val_dataset = SyntheticSmokeDataset(
            K, R, t, n_frames=30, n_joints=n_joints, clip_len=args.clip_len
        )
        n_views = 4
    else:
        train_datasets = [RandomClipDataset(tp, args.clip_len, n_samples=args.train_samples) for tp in args.train]
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        val_dataset = TemporalClipDataset(args.val, args.clip_len, stride=args.val_stride)
        sample = np.load(args.train[0])
        n_views = int(sample["camera_K"].shape[0])
        n_joints = int(sample["points_2d"].shape[2])
        K_sample = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    args.j = n_joints
    args.n_views = n_views

    model = OmniMultiViewFusionV3(
        j=n_joints,
        d=args.d,
        n_views=n_views,
        n_heads=args.n_heads,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        graph_num_layers=args.graph_num_layers,
        n_joint_layers=args.n_joint_layers,
        epipolar_loss_weight=args.epipolar_loss_weight,
        return_pp_delta=False,
        return_covariance=True,
        use_multiscale_fusion=args.use_multiscale_fusion,
        use_camera_conditioning=args.use_camera_conditioning,
        use_epipolar_bias=args.use_epipolar_bias,
        multiscale_scales=args.multiscale_scales,
        camera_condition_dim=args.camera_condition_dim,
        epipolar_temperature=args.epipolar_temperature,
    ).to(device)

    if n_joints != 17:
        model.rebuild_graph(n_joints, dataset="mpiinf3dhp")

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    if args.warm_start is not None:
        load_warm_start(model, args.warm_start)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    total_epochs = args.epochs
    warmup_epochs = args.lr_warmup_epochs if args.lr_cosine else 0
    scheduler = build_lr_scheduler(optimizer, total_epochs, warmup_epochs, args.lr_min) if args.lr_cosine else None

    ema_decay = args.ema_decay if args.ema_decay > 0.0 else None

    trainer = OmniMultiViewTrainer(
        model,
        optimizer,
        device,
        args=args,
        total_epochs=total_epochs,
        max_grad_norm=args.max_grad_norm,
        amp_enabled=args.amp,
        ema_decay=ema_decay,
        scheduler=scheduler,
    )

    freeze_epochs = args.warm_start_freeze_epochs if args.warm_start else 0
    if freeze_epochs > 0 and freeze_epochs >= total_epochs:
        print("warm_start_freeze_epochs >= total_epochs; no end-to-end training will occur.")
        freeze_epochs = max(0, total_epochs - 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if freeze_epochs > 0:
        print(f"Freezing encoder / ST-transformer for {freeze_epochs} epoch(s)...")
        freeze_old_params(model)
        for freeze_i in range(freeze_epochs):
            trainer.epoch += 1
            train_metrics = trainer.train_epoch(train_loader)
            val_metrics = trainer.evaluate(val_loader, compute_metric=build_eval_metric())
            trainer.step_scheduler()
            print(
                f"[Freeze] Epoch {trainer.epoch}: train_loss={train_metrics.get('loss', float('nan')):.6f}, "
                f"val_MPJPE={val_metrics.get('mpjpe', float('nan')) * 1000:.2f}mm"
            )
        print("Unfreezing full model for end-to-end training.")
        unfreeze_all(model)

    remaining_epochs = total_epochs - freeze_epochs
    if remaining_epochs > 0:
        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=remaining_epochs,
            eval_metric=build_eval_metric(),
            checkpoint_path=str(output_path),
            save_best=True,
        )

        best = min(history, key=lambda e: e.get("val", {}).get("loss", float("inf")))
        best_mpjpe = best.get("val", {}).get("mpjpe", float("nan"))
        print(f"MPJPE at best val loss: {best_mpjpe * 1000:.2f}mm -> {output_path}")

    trainer.save_checkpoint(str(output_path.with_suffix("")) + "_final.pth")


if __name__ == "__main__":
    main()
