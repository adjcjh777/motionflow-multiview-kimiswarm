"""Train OmniMultiViewFusionV5 on one or more WebBridge datasets.

This script mirrors ``train_omniview_fusion_v4_webbridge_multi.py`` but targets
``OmniMultiViewFusionV5``.  It adds v5-specific capabilities:

* camera-conditioned view embeddings (permutation invariant over view order),
* set-transformer view aggregation,
* variable-view training with random view subsets and permutations.

Usage
-----
    # CPU smoke (1 epoch, synthetic data, no external files)
    python experiments/train_omniview_fusion_v5_webbridge_multi.py --smoke

    # Full training on a WebBridge manifest
    python experiments/train_omniview_fusion_v5_webbridge_multi.py \
        --manifest configs/splits/webbridge_h36m_train_val.yaml \
        --use_camera_view_embedding --use_set_view_aggregator \
        --d 128 --residual_hidden 128 --epochs 30
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.perturb import perturb_cameras  # noqa: E402
from motionflow_mv.data.split_loader import (  # noqa: E402
    load_multi_dataset_manifest,
    load_split_manifest,
)
from motionflow_mv.data.webbridge_mixed_dataset import (  # noqa: E402
    build_webbridge_mixed_dataloaders,
    webbridge_mixed_collate_fn,
)
from motionflow_mv.fusion.prototypes.cross_view_graph_attention import (  # noqa: E402
    H36M_17_PARENTS,
    MPI_INF_3DHP_28_PARENTS,
)
from motionflow_mv.fusion.omniview_fusion_v5 import (  # noqa: E402
    OmniMultiViewFusionV5,
)
from motionflow_mv.training.trainer_v2 import TrainerV2, build_lr_scheduler  # noqa: E402


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
# Data loading (same pattern as train_omniview_fusion_v4_webbridge_multi.py)
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
    variable_view_subset: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Lightweight per-clip augmentation with optional view dropout.

    When ``variable_view_subset`` is True, instead of independent per-view
    dropout we sample a random subset size ``k ~ Uniform(min_views, V)``
    and keep exactly ``k`` views per clip. This forces the model to learn
    from arbitrary view cardinalities.
    """
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = (torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device) > dropout_rate).float()
        x[..., 2] = x[..., 2] * mask

    view_mask = None
    if variable_view_subset:
        B = x.shape[0]
        V = x.shape[2]
        view_mask = torch.zeros(B, V, device=x.device)
        for i in range(B):
            k = torch.randint(min_views, V + 1, (1,)).item()
            idx = torch.randperm(V)[:k]
            view_mask[i, idx] = 1.0
        x[..., 2] = x[..., 2] * view_mask.view(B, 1, V, 1)
    elif view_dropout_rate > 0:
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

    return x, view_mask


# ---------------------------------------------------------------------------
# Calibration perturbation helpers
# ---------------------------------------------------------------------------

def _camera_perturbation_schedule(epoch: int, args: Namespace) -> Dict[str, float]:
    """Return the per-epoch camera perturbation standard deviations."""
    try:
        from motionflow_mv.calibration.camera_perturbation_curriculum import (  # noqa: E402
            extended_camera_perturbation_schedule_with_anneal,
        )

        return extended_camera_perturbation_schedule_with_anneal(
            epoch,
            total_epochs=args.epochs,
            schedule=args.cam_aug_schedule,
            rot=args.cam_aug_rot,
            trans=args.cam_aug_trans,
            focal=args.cam_aug_focal,
            pp=args.cam_aug_pp,
            ramp_epochs=args.cam_aug_ramp_epochs,
            intrinsics_ramp_epochs=args.cam_aug_intrinsics_ramp_epochs,
            warmup_epochs=args.cam_aug_warmup_epochs,
        )
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Camera curriculum unavailable ({exc}); using flat perturbations.")
        return {
            "rot_std": args.cam_aug_rot,
            "trans_std": args.cam_aug_trans,
            "focal_std": args.cam_aug_focal,
            "pp_std": args.cam_aug_pp,
        }


def apply_calibration_perturbation(
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    epoch: int,
    args: Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply epoch-dependent calibration perturbations to cameras."""
    if args.cam_aug_schedule == "none":
        return K, R, t
    stds = _camera_perturbation_schedule(epoch, args)
    return perturb_cameras(
        K,
        R,
        t,
        rot_std=stds["rot_std"],
        trans_std=stds["trans_std"],
        focal_std=stds["focal_std"],
        pp_std=stds["pp_std"],
    )


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


def attention_entropy_loss(weights: torch.Tensor, dim: int = -3) -> torch.Tensor:
    """Entropy regularisation on per-view triangulation weights."""
    p = weights / (weights.sum(dim=dim, keepdim=True) + 1e-8)
    p = p.clamp(min=1e-8)
    entropy = -(p * torch.log(p)).sum(dim=dim).mean()
    return entropy


def budget_loss(weights: torch.Tensor, target_k: float) -> torch.Tensor:
    """Mean-squared deviation between the number of active views and ``target_k``."""
    active = (weights > 0.1).float().sum(dim=-2).mean()
    return F.mse_loss(active, torch.tensor(target_k, device=weights.device, dtype=weights.dtype))


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
        # Mixed loader returns (x, y, K, R, t, dataset_id); the dataset_id is used
        # to build a base view mask for padded views.
        if len(batch) == 6:
            x, y, K, R, t, dataset_id = batch
            dataset_id = dataset_id.to(device)
        else:
            x, y, K, R, t = batch
            dataset_id = None

        x = x.to(device)
        y = y.to(device)
        K = K.to(device)
        R = R.to(device)
        t = t.to(device)

        # Training-only augmentations.  Operate on a clone so the DataLoader
        # buffer is not silently modified.
        x, view_mask = augment_clip(
            x.clone(),
            noise_std=args.noise_std if not args.smoke else 0.0,
            dropout_rate=args.confidence_dropout,
            view_dropout_rate=args.view_dropout_rate,
            min_views=args.min_views,
            variable_view_subset=args.variable_view_subset,
        )

        # When using the mixed loader, padded views must be masked out.  The
        # dataset_id encodes the source domain (0=H36M/4 views, 1=MPI/14 views).
        if dataset_id is not None:
            B, T, V_full = x.shape[0], x.shape[1], x.shape[2]
            base_view_mask = torch.zeros(B, T, V_full, device=device)
            for i in range(B):
                if dataset_id[i].item() == 0:  # H36M
                    base_view_mask[i, :, :4] = 1.0
                elif dataset_id[i].item() == 1:  # MPI-INF-3DHP
                    base_view_mask[i, :, :14] = 1.0
                else:
                    base_view_mask[i, :, :] = 1.0
            if view_mask is None:
                view_mask = base_view_mask
            else:
                view_mask = view_mask * base_view_mask

        # Optional variable-view training: sample a random subset of views and
        # optionally permute the order of all views.  The model still receives
        # all V camera tensors (so internal buffers sized for ``n_views`` stay
        # valid), but ``view_mask`` masks out the unselected views, giving a
        # variable effective number of actual views per batch.
        if args.use_variable_view_training:
            B, T, V_full = x.shape[0], x.shape[1], x.shape[2]
            k = torch.randint(
                args.variable_view_min_views,
                args.variable_view_max_views + 1,
                (1,),
            ).item()
            k = min(k, V_full)
            view_mask = torch.zeros(B, T, V_full, device=device)
            for i in range(B):
                selected = torch.randperm(V_full)[:k]
                view_mask[i, :, selected] = 1.0

            if args.variable_view_permute:
                perm = torch.randperm(V_full)
                x = x[:, :, perm, :, :]
                K = K[:, perm, :, :]
                R = R[:, perm, :, :]
                t = t[:, perm, :]
                view_mask = view_mask[:, :, perm]

        # Optional calibration curriculum.
        K_aug, R_aug, t_aug = apply_calibration_perturbation(
            K, R, t, model.epoch if hasattr(model, "epoch") else 0, args
        )

        if view_mask is not None:
            out = model(x, K=K_aug, R=R_aug, t=t_aug, view_mask=view_mask)
        else:
            out = model(x, K=K_aug, R=R_aug, t=t_aug)
        pred_3d = out[0]
        weights = out[1]
        visibility = out[2]
        L = out[3]
        epi_loss = out[4]
        entropy_loss_out = out[5] if len(out) > 5 else None
        budget_loss_out = out[6] if len(out) > 6 else None

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
            nll = uncertainty_nll_loss(pred_3d, x[..., :2], x[..., 2], K_aug, R_aug, t_aug, L)
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

        if args.attention_entropy_weight > 0.0:
            if entropy_loss_out is not None:
                loss = loss + args.attention_entropy_weight * entropy_loss_out
                metrics["entropy_loss"] = entropy_loss_out.item()
            else:
                ent = attention_entropy_loss(weights)
                loss = loss + args.attention_entropy_weight * ent
                metrics["entropy_loss"] = ent.item()

        if args.budget_loss_weight > 0.0:
            if budget_loss_out is not None:
                loss = loss + args.budget_loss_weight * budget_loss_out
                metrics["budget_loss"] = budget_loss_out.item()
            elif args.adaptive_view_k is not None and args.adaptive_view_k > 0:
                bgt = budget_loss(weights, args.adaptive_view_k)
                loss = loss + args.budget_loss_weight * bgt
                metrics["budget_loss"] = bgt.item()

        # Optional monotonic multi-view loss: error with a subset of views should
        # not be better than using all views.  This is only meaningful when the
        # model is trained with variable-view subsets.
        if args.monotonic_loss_weight > 0.0 and view_mask is not None:
            with torch.no_grad():
                out_full = model(x, K=K_aug, R=R_aug, t=t_aug, view_mask=None)
            pred_full = out_full[0]
            err_subset = F.mse_loss(pred_3d, y)
            err_full = F.mse_loss(pred_full, y)
            active = view_mask.sum(dim=(1, 2)).max().item() / max(1, view_mask.shape[1])
            k_active = max(2, int(round(active)))
            margin = (args.monotonic_margin / 1000.0) / max(1, k_active)
            mono = torch.clamp(err_subset - err_full - margin, min=0.0)
            loss = loss + args.monotonic_loss_weight * mono
            metrics["monotonic_loss"] = mono.item()

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
    """TrainerV2 pre-wired with the OmniMultiViewFusionV5 loss mix."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        args: Namespace,
        **kwargs: Any,
    ):
        self.args = args
        compute_loss = build_compute_loss(args)
        super().__init__(
            model,
            optimizer,
            device,
            compute_loss=compute_loss,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Multi-dataset helpers
# ---------------------------------------------------------------------------

def _inspect_npz(npz_path: str) -> Tuple[int, int]:
    """Return (n_views, n_joints) for a canonical .npz file."""
    data = np.load(npz_path)
    points_2d = data["points_2d"]
    return int(points_2d.shape[1]), int(points_2d.shape[2])


def _validate_dataset_consistency(
    train_files: List[str],
    val_files: List[str],
    graph_num_layers: int,
) -> Tuple[int, int]:
    """Ensure all files share the same (n_views, n_joints)."""
    all_files = list(train_files) + list(val_files)
    if not all_files:
        raise ValueError("No train or validation files found in the provided manifests.")

    shapes = []
    for f in all_files:
        if not Path(f).exists():
            raise FileNotFoundError(f"Manifest references missing file: {f}")
        shapes.append(_inspect_npz(f))

    unique = set(shapes)
    if len(unique) != 1:
        from collections import defaultdict
        grouped: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for f, s in zip(all_files, shapes):
            grouped[s].append(f)
        summary = "\n".join(
            f"  {s[1]} joints / {s[0]} views: {len(paths)} file(s)"
            for s, paths in sorted(grouped.items())
        )
        raise ValueError(
            "Mixed (n_views, n_joints) configurations detected across the selected manifests. "
            "OmniMultiViewFusionV5 is instantiated once per run and therefore requires all "
            "selected files to share the same view count and skeleton.\n"
            f"Detected configurations:\n{summary}\n"
            "Recommended fix: pass only manifests that share the same (n_views, n_joints), "
            "e.g. train on MPI-INF-3DHP only, or H36M only."
        )

    n_views, n_joints = shapes[0]

    if graph_num_layers > 0 and n_joints not in (17, 28):
        raise ValueError(
            f"OmniMultiViewFusionV5 graph attention only supports 17 or 28 joints, "
            f"but the selected data has {n_joints} joints. "
            f"Use --graph_num_layers 0 to disable graph attention for this skeleton."
        )

    return n_views, n_joints


def build_datasets(args: Namespace) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int, int]:
    """Build train/val ConcatDatasets from manifest(s) and infer (n_views, n_joints)."""
    if args.smoke:
        K, R, t = _make_synthetic_cameras(n_views=4)
        n_joints = 17
        train_dataset: torch.utils.data.Dataset = SyntheticSmokeDataset(
            K, R, t, n_frames=50, n_joints=n_joints, clip_len=args.clip_len
        )
        val_dataset: torch.utils.data.Dataset = SyntheticSmokeDataset(
            K, R, t, n_frames=30, n_joints=n_joints, clip_len=args.clip_len
        )
        return train_dataset, val_dataset, 4, n_joints

    if args.use_mixed_loader:
        if args.mixed_manifest is None:
            raise ValueError("--mixed_manifest is required when --use_mixed_loader is set")
        import yaml

        with open(args.mixed_manifest, "r") as f:
            mixed_cfg = yaml.safe_load(f)

        train_paths = mixed_cfg["train_paths"]
        train_names = mixed_cfg["train_names"]
        val_paths = mixed_cfg["val_paths"]
        val_names = mixed_cfg["val_names"]
        if not (len(train_paths) == len(train_names) and len(val_paths) == len(val_names)):
            raise ValueError("Mixed manifest: paths and names must have the same length")

        train_loader, val_loader = build_webbridge_mixed_dataloaders(
            train_paths=train_paths,
            train_names=train_names,
            val_paths=val_paths,
            val_names=val_names,
            clip_len=args.clip_len,
            batch_size=args.batch_size,
            train_samples=args.train_samples,
            val_stride=args.val_stride,
            num_workers=0,
        )
        # The mixed loader always pads to 14 views and maps to the 17-joint skeleton.
        return train_loader.dataset, val_loader.dataset, 14, 17

    if args.manifest:
        split = load_multi_dataset_manifest(args.manifest)
    else:
        split = {"train": args.train or [], "val": args.val or [], "test": []}

    train_files = split.get("train", [])
    val_files = split.get("val", [])

    if not train_files or not val_files:
        raise ValueError(
            "No train/val files found. Provide --manifest or both --train and --val."
        )

    n_views, n_joints = _validate_dataset_consistency(train_files, val_files, args.graph_num_layers)

    train_datasets = [RandomClipDataset(tp, args.clip_len, n_samples=args.train_samples) for tp in train_files]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = TemporalClipDataset(val_files[0], args.clip_len, stride=args.val_stride)

    if len(val_files) > 1:
        val_datasets = [TemporalClipDataset(vp, args.clip_len, stride=args.val_stride) for vp in val_files]
        val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    return train_dataset, val_dataset, n_views, n_joints


# ---------------------------------------------------------------------------
# Warm-start helpers
# ---------------------------------------------------------------------------

def load_warm_start(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Load a checkpoint, tolerating extra/missing keys for warm-starting."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warm-start: missing keys (expected for new heads): {missing[:10]}")
    if unexpected:
        print(f"Warm-start: unexpected keys ignored: {unexpected[:10]}")
    print(f"Warm-started from {checkpoint_path}")


def freeze_old_params(model: torch.nn.Module) -> None:
    """Freeze everything except the new v5-specific heads."""
    new_prefixes = (
        "graph_joint_attention",
        "visibility_head",
        "omni_joint_attn",
        "multiscale_fusion",
        "camera_conditioning",
        "rotation_correction",
        "adaptive_view_selector",
        "kinematic_refiner",
        "skeleton_graph_residual",
        "camera_view_embedding",
        "set_view_aggregator",
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
        description="Train OmniMultiViewFusionV5 on one or more WebBridge datasets",
    )
    # Data
    parser.add_argument(
        "--manifest",
        type=str,
        action="append",
        default=None,
        help="Path to a YAML split manifest (can be passed multiple times). "
             "Default: configs/splits/webbridge_all_train.yaml",
    )
    parser.add_argument("--train", type=str, nargs="+", default=None, help="Train .npz files (legacy, overrides manifest train)")
    parser.add_argument("--val", type=str, default=None, help="Validation .npz file (legacy, overrides manifest val)")
    parser.add_argument("--smoke", action="store_true", help="1-epoch CPU/GPU smoke test on synthetic data")
    parser.add_argument("--use_mixed_loader", action="store_true", help="Use the WebBridge mixed-dataset loader (H36M+MPI, 17 joints/14 views)")
    parser.add_argument("--mixed_manifest", type=str, default=None, help="YAML manifest for mixed loader (train_paths/train_names/val_paths/val_names)")
    # Model
    parser.add_argument("--d", type=int, default=128, help="Model feature dimension")
    parser.add_argument("--residual_hidden", type=int, default=128, help="Residual MLP hidden size")
    parser.add_argument("--n_st_layers", type=int, default=2, help="Spatio-temporal transformer layers")
    parser.add_argument("--graph_num_layers", type=int, default=1, help="Graph-joint attention layers")
    parser.add_argument("--n_joint_layers", type=int, default=0, help="Dense joint-level transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--epipolar_loss_weight", type=float, default=0.05, help="Epipolar loss weight passed to the model")
    # v4 toggles
    parser.add_argument("--use_multiscale_fusion", type=lambda x: x.lower() == "true", default=True, help="Enable hierarchical multi-scale fusion")
    parser.add_argument("--use_camera_conditioning", type=lambda x: x.lower() == "true", default=True, help="Enable camera conditioning")
    parser.add_argument("--use_epipolar_bias", type=lambda x: x.lower() == "true", default=True, help="Enable epipolar-biased ST transformer")
    parser.add_argument("--use_context_visibility", type=lambda x: x.lower() == "true", default=True, help="Use context-aware visibility head")
    parser.add_argument("--use_skeleton_residual", type=lambda x: x.lower() == "true", default=True, help="Use skeleton-graph residual refiner")
    parser.add_argument("--use_kinematic_refiner", type=lambda x: x.lower() == "true", default=False, help="Use kinematic-chain final refiner")
    parser.add_argument("--use_adaptive_view_selection", type=lambda x: x.lower() == "true", default=False, help="Use adaptive view selection")
    parser.add_argument("--use_rotation_correction", type=lambda x: x.lower() == "true", default=False, help="Use rotation correction head")
    parser.add_argument("--use_entropy_regularization", type=lambda x: x.lower() == "true", default=False, help="Enable entropy regularisation inside the model")
    parser.add_argument("--adaptive_view_k", type=int, default=None, help="Target k for adaptive view selection")
    # v5 toggles
    parser.add_argument("--use_camera_view_embedding", action="store_true", help="Use camera-conditioned view embedding instead of learned view_pos_embed")
    parser.add_argument("--use_set_view_aggregator", action="store_true", help="Use set-transformer (ISAB) view aggregator before the time+view transformer")
    parser.add_argument("--camera_view_embedding_hidden", type=int, default=32, help="Hidden dimension of camera-conditioned view embedding MLP")
    parser.add_argument("--set_view_n_isab_layers", type=int, default=2, help="Number of ISAB layers in set aggregator")
    parser.add_argument("--set_view_num_inducing_points", type=int, default=32, help="Number of inducing points per ISAB")
    parser.add_argument("--set_view_dropout", type=float, default=0.0, help="Dropout in set aggregator")
    # v6 toggles
    parser.add_argument("--use_perceiver_aggregator", action="store_true", help="Use Perceiver-style view aggregator instead of ISAB")
    parser.add_argument("--perceiver_n_latents", type=int, default=16, help="Number of latent vectors in Perceiver aggregator")
    parser.add_argument("--perceiver_n_layers", type=int, default=2, help="Number of Perceiver latent layers")
    parser.add_argument("--perceiver_n_heads", type=int, default=4, help="Number of attention heads in Perceiver aggregator")
    parser.add_argument("--perceiver_dropout", type=float, default=0.0, help="Dropout in Perceiver aggregator")
    parser.add_argument("--monotonic_loss_weight", type=float, default=0.0, help="Weight for monotonic multi-view ranking loss")
    parser.add_argument("--monotonic_margin", type=float, default=5.0, help="Margin (mm) for monotonic multi-view loss")
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
    parser.add_argument("--variable_view_subset", action="store_true", help="Train with random view-subset sampling (k ~ Uniform(min_views, V))")
    parser.add_argument("--use_variable_view_training", action="store_true", help="Randomly sample and permute view subsets each batch")
    parser.add_argument("--variable_view_min_views", type=int, default=2, help="Minimum views in variable-view training subset")
    parser.add_argument("--variable_view_max_views", type=int, default=4, help="Maximum views in variable-view training subset")
    parser.add_argument("--variable_view_permute", action="store_true", help="Permute selected view order during variable-view training")
    # Calibration curriculum
    parser.add_argument("--cam_aug_schedule", type=str, default="none", choices=["none", "extended_curriculum", "extended_intrinsics_curriculum"], help="Camera perturbation schedule")
    parser.add_argument("--cam_aug_rot", type=float, default=0.5, help="Rotation perturbation std (degrees)")
    parser.add_argument("--cam_aug_trans", type=float, default=0.005, help="Translation perturbation std")
    parser.add_argument("--cam_aug_focal", type=float, default=0.01, help="Focal-length perturbation std (fraction)")
    parser.add_argument("--cam_aug_pp", type=float, default=2.0, help="Principal-point perturbation std (pixels)")
    parser.add_argument("--cam_aug_ramp_epochs", type=int, default=10, help="Epochs to ramp extrinsics")
    parser.add_argument("--cam_aug_intrinsics_ramp_epochs", type=int, default=5, help="Epochs to ramp intrinsics")
    parser.add_argument("--cam_aug_warmup_epochs", type=int, default=0, help="Warmup epochs before ramping")
    # Loss weights
    parser.add_argument("--visibility_loss_weight", type=float, default=0.1, help="Visibility BCE weight")
    parser.add_argument("--uncertainty_loss_weight", type=float, default=0.05, help="Uncertainty NLL weight")
    parser.add_argument("--temporal_loss_weight", type=float, default=0.02, help="Temporal consistency weight")
    parser.add_argument("--bone_loss_weight", type=float, default=0.05, help="Bone-length consistency weight")
    parser.add_argument("--attention_entropy_weight", type=float, default=0.0, help="Attention-entropy regularisation weight")
    parser.add_argument("--budget_loss_weight", type=float, default=0.0, help="Adaptive-view budget loss weight")
    # Warm-start
    parser.add_argument("--warm_start", type=str, default=None, help="Path to v2/v3 checkpoint for warm-starting")
    parser.add_argument("--warm_start_freeze_epochs", type=int, default=0, help="Freeze encoder/transformer for N epochs after warm-start")
    # I/O
    parser.add_argument("--output", type=str, default="outputs/omniview_fusion_v5_webbridge_multi.pth", help="Checkpoint path")
    args = parser.parse_args()

    # Default manifest if none provided and no legacy --train/--val.
    if args.manifest is None and (args.train is None or args.val is None):
        args.manifest = ["configs/splits/webbridge_all_train.yaml"]

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

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_dataset, val_dataset, n_views, n_joints = build_datasets(args)

    selected_collate_fn = webbridge_mixed_collate_fn if args.use_mixed_loader else collate_fn
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=selected_collate_fn,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=selected_collate_fn,
        num_workers=0,
    )

    args.j = n_joints
    args.n_views = n_views

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_kwargs = {
        "j": n_joints,
        "d": args.d,
        "n_views": n_views,
        "n_heads": args.n_heads,
        "n_joint_layers": args.n_joint_layers,
        "n_st_layers": args.n_st_layers,
        "residual_hidden": args.residual_hidden,
        "graph_num_layers": args.graph_num_layers,
        "epipolar_loss_weight": args.epipolar_loss_weight,
        "return_pp_delta": False,
        "return_covariance": True,
        # v4 toggles
        "use_multiscale_fusion": args.use_multiscale_fusion,
        "use_camera_conditioning": args.use_camera_conditioning,
        "use_epipolar_bias": args.use_epipolar_bias,
        "use_context_visibility": args.use_context_visibility,
        "use_skeleton_residual": args.use_skeleton_residual,
        "use_kinematic_refiner": args.use_kinematic_refiner,
        "use_adaptive_view_selection": args.use_adaptive_view_selection,
        "use_rotation_correction": args.use_rotation_correction,
        "use_entropy_regularization": args.use_entropy_regularization,
        # v5 toggles
        "use_camera_view_embedding": args.use_camera_view_embedding,
        "use_set_view_aggregator": args.use_set_view_aggregator,
        "camera_view_embedding_hidden": args.camera_view_embedding_hidden,
        "set_view_n_isab_layers": args.set_view_n_isab_layers,
        "set_view_num_inducing_points": args.set_view_num_inducing_points,
        "set_view_dropout": args.set_view_dropout,
        # v6 toggles
        "use_perceiver_aggregator": args.use_perceiver_aggregator,
        "perceiver_n_latents": args.perceiver_n_latents,
        "perceiver_n_layers": args.perceiver_n_layers,
        "perceiver_n_heads": args.perceiver_n_heads,
        "perceiver_dropout": args.perceiver_dropout,
    }
    if args.adaptive_view_k is not None:
        model_kwargs["adaptive_view_target_k"] = args.adaptive_view_k

    model = OmniMultiViewFusionV5(**model_kwargs).to(device)

    if n_joints != 17:
        if n_joints == 28 and hasattr(model, "rebuild_graph"):
            model.rebuild_graph(n_joints, dataset="mpiinf3dhp")

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    # ------------------------------------------------------------------
    # Warm-start
    # ------------------------------------------------------------------
    if args.warm_start is not None:
        load_warm_start(model, args.warm_start)

    # ------------------------------------------------------------------
    # Optimizer / scheduler / trainer
    # ------------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    total_epochs = args.epochs
    warmup_epochs = args.lr_warmup_epochs if args.lr_cosine else 0
    scheduler = build_lr_scheduler(optimizer, total_epochs, warmup_epochs, args.lr_min) if args.lr_cosine else None

    ema_decay = args.ema_decay if args.ema_decay > 0.0 else None

    trainer = OmniMultiViewTrainer(
        model,
        optimizer,
        device,
        args,
        total_epochs=total_epochs,
        max_grad_norm=args.max_grad_norm,
        amp_enabled=args.amp,
        ema_decay=ema_decay,
        scheduler=scheduler,
    )

    # ------------------------------------------------------------------
    # Optional staged warm-start: freeze old params for a few epochs.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
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
        print(f"Best val MPJPE: {best_mpjpe * 1000:.2f}mm -> {output_path}")

    # Always save the final checkpoint as well.
    final_path = str(output_path.with_suffix("")) + "_final.pth"
    trainer.save_checkpoint(final_path)

    # Save the training configuration so the eval script can reconstruct the model
    # with the exact flags used during training.
    config_path = output_path.with_suffix(".config.json")
    try:
        import json

        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Saved training config -> {config_path}")
    except Exception as exc:
        print(f"Warning: could not save training config: {exc}")


if __name__ == "__main__":
    main()
