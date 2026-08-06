"""Train RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint on MPI-INF-3DHP.

This script is a thin wrapper around ``train_ray_attention_temporal_crossview_residual_mpiinf3dhp.py``
that swaps the model for ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint``, which
adds a bounded, learned principal-point correction layer before the cross-view spatio-temporal
attention and differentiable triangulation step.  Camera perturbations during training force the
principal-point head to learn a meaningful correction.

Usage
-----
    conda run -n mf python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
        --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
               data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 --epochs 10 \
        --pp_loss_weight 0.1 --cam_aug_pp 5.0 --cam_aug_focal 0.01
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.perturb import perturb_cameras_with_delta
from motionflow_mv.fusion.ray_attention_temporal_crossview_factorized_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewFactorizedResidualPrincipalPoint,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from motionflow_mv.losses import reprojection_loss, velocity_loss
from motionflow_mv.losses.reprojection_consistency import robust_reprojection_loss
from motionflow_mv.losses.view_selection_loss import ViewSelectionLoss
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_dynamic_gate_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointDynamicGate,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_graph_skeleton_residual_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraphSkeletonResidual,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_epipolar_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolar,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_splat_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSplat,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_kinematic_chain_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointKinematicChain,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_crossview_contrast_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCrossViewContrast,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBayesianTri,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_epipolar_bias_v2_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_camera_conditioned_model import (
    RayAttentionFusionModelTemporalCrossviewResidualCameraConditioned,
)
from motionflow_mv.fusion.ray_attention_hierarchical_view_temporal_joint_residual_principal_point_model import (
    RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_deeper_temporal_model import (
    RayAttentionFusionModelHierarchicalViewDeeperTemporalResidualPrincipalPoint,
)
from motionflow_mv.losses.gaussian_splatting_pose_loss import gaussian_splatting_pose_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end],
             self.confidences[start:end].unsqueeze(-1)],
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

    def __getitem__(self, idx):
        start = random.randint(0, max(0, self.total_frames - self.clip_len))
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end],
             self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def augment_clip(x, noise_std: float = 0.5, dropout_rate: float = 0.1,
                 outlier_rate: float = 0.02, outlier_scale: float = 100.0,
                 view_dropout_rate: float = 0.0, min_views: int = 2,
                 view_noise_std: float = 0.0, joint_dropout_rate: float = 0.0):
    """Lightweight per-clip augmentation."""
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if view_noise_std > 0:
        # Per-view independent 2-D Gaussian noise.
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * view_noise_std
    if dropout_rate > 0:
        mask = (torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device) > dropout_rate).float()
        x[..., 2] = x[..., 2] * mask
    if outlier_rate > 0:
        outlier_mask = torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device) < outlier_rate
        outlier = (torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], 2, device=x.device) - 0.5) * 2 * outlier_scale
        x[..., :2] = torch.where(outlier_mask[..., None], outlier, x[..., :2])
    if view_dropout_rate > 0:
        B = x.shape[0]
        V = x.shape[2]
        view_mask = (torch.rand(B, V, device=x.device) > view_dropout_rate).float()  # 1 = keep
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
    if joint_dropout_rate > 0:
        # Randomly zero-out per-joint confidence per view.
        joint_mask = (torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device) > joint_dropout_rate).float()
        x[..., 2] = x[..., 2] * joint_mask
    return x


def evaluate(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, *_ = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            total_err += err.item() * xb.size(0)
            total_count += xb.size(0)
    return total_err / total_count


def main():
    parser = argparse.ArgumentParser(description="Train temporal + cross-view ray-attention fusion with residual refinement and principal-point correction on MPI-INF-3DHP")
    parser.add_argument("--train", type=str, nargs="+", required=True, help="Train .npz files")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--model_type", type=str, default="temporal", choices=["temporal", "factorized", "dynamic_gate", "graph_skeleton_residual", "epipolar", "epipolar_bias_v2_pp", "epipolar_bias_v2_lite_pp", "splat", "kinematic_chain", "crossview_contrast", "bayesian_tri", "camera_conditioned_pp", "hierarchical_view_temporal_joint_pp", "deeper_temporal_pp"], help="Backbone type: temporal (time+view), factorized (alternating view/temporal), dynamic_gate (anchor + per-view gate), graph_skeleton_residual (skeleton-graph residual refiner), epipolar (epipolar-biased weight head), epipolar_bias_v2_pp (epipolar-biased ST transformer v2), epipolar_bias_v2_lite_pp (late-layer epipolar-biased ST transformer v2 lite), splat (Gaussian-splatting pose regularizer), kinematic_chain (kinematic-chain graph refiner), crossview_contrast (cross-view contrastive pose representation), bayesian_tri (uncertainty-aware triangulation with adaptive Gauss-Newton), camera_conditioned_pp (camera-parameter-conditioned weight + residual heads), hierarchical_view_temporal_joint_pp (hierarchical view -> temporal -> skeleton-joint attention), or deeper_temporal_pp (hierarchical view -> deeper residual-gated temporal -> skeleton-joint attention)")
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--n_view_layers", type=int, default=2)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--n_view_groups", type=int, default=2, help="Number of camera groups for hierarchical_view_temporal_joint_pp")
    parser.add_argument("--n_joint_graph_layers", type=int, default=1, help="Number of skeleton-graph layers for hierarchical_view_temporal_joint_pp")
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--gate_sparsity_weight", type=float, default=0.0, help="Sparsity regulariser weight for the dynamic view gate")
    parser.add_argument("--gate_entropy_weight", type=float, default=0.0, help="Entropy regulariser weight for the dynamic view gate")
    parser.add_argument("--view_noise_std", type=float, default=0.0, help="Per-view 2D Gaussian noise std (pixels) for dynamic_gate")
    parser.add_argument("--joint_dropout_rate", type=float, default=0.0, help="Per-joint confidence dropout rate for dynamic_gate")
    parser.add_argument("--principal_point_hidden", type=int, default=64)
    parser.add_argument("--principal_point_max_offset", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=4000, help="Random clips per train sequence")
    parser.add_argument("--val_stride", type=int, default=1, help="Stride for validation clips (higher = faster)")
    parser.add_argument("--velocity_loss_weight", type=float, default=0.0, help="Weight for temporal velocity consistency auxiliary loss (requires clip_len >= 2)")
    parser.add_argument("--reproj_weight", type=float, default=0.0, help="Weight for reprojection auxiliary loss")
    parser.add_argument("--reproj_raw_weight", type=float, default=0.0, help="Weight for robust reprojection loss on the raw triangulated pose")
    parser.add_argument("--reproj_refined_weight", type=float, default=0.0, help="Weight for robust reprojection loss on the refined pose")
    parser.add_argument("--reproj_robust", action="store_true", help="Use Charbonnier robust reprojection loss instead of MSE")
    parser.add_argument("--reproj_mask_dropout", action="store_true", help="Mask zero-confidence joints in reprojection loss")
    parser.add_argument("--return_raw_3d", action="store_true", help="Return raw triangulated 3D pose for reprojection supervision")
    parser.add_argument("--pp_loss_weight", type=float, default=0.0, help="Weight for principal-point offset supervision loss")
    parser.add_argument("--focal_loss_weight", type=float, default=None, help="Weight for focal scale supervision loss (defaults to pp_loss_weight)")
    parser.add_argument("--focal_max_scale", type=float, default=0.0, help="Maximum predicted focal-length scale; 0 disables focal correction")
    parser.add_argument("--cam_aug_rot", type=float, default=0.5, help="Camera rotation augmentation std in degrees")
    parser.add_argument("--cam_aug_trans", type=float, default=0.005, help="Camera translation augmentation std in meters")
    parser.add_argument("--cam_aug_focal", type=float, default=0.01, help="Camera focal length augmentation std (relative)")
    parser.add_argument("--cam_aug_pp", type=float, default=2.0, help="Camera principal point augmentation std in pixels")
    parser.add_argument("--cam_aug_schedule", type=str, default="flat", choices=["flat", "extrinsic_curriculum", "intrinsics_curriculum"], help="Camera augmentation schedule")
    parser.add_argument("--cam_aug_ramp_epochs", type=int, default=10, help="Number of epochs over which to ramp extrinsic augmentation for extrinsic_curriculum")
    parser.add_argument("--cam_aug_intrinsics_ramp_epochs", type=int, default=5, help="Number of epochs over which to ramp PP/focal augmentation for intrinsics_curriculum")
    parser.add_argument("--view_dropout_rate", type=float, default=0.0, help="Probability of dropping an entire camera view during training (0 disables)")
    parser.add_argument("--min_views", type=int, default=2, help="Minimum number of views kept when view_dropout_rate > 0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm_start", type=str, default=None, help="Path to checkpoint to warm-start from (loads state_dict with strict=False)")
    parser.add_argument("--output", type=str, default="outputs/ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.pth")
    parser.add_argument("--pp_pretrain_epochs", type=int, default=0, help="Number of initial epochs to train only the principal_point_correction head")
    parser.add_argument("--splat_loss_weight", type=float, default=0.0, help="Weight for Gaussian-splatting pose regularizer loss (splat model only)")
    parser.add_argument("--epipolar_loss_weight", type=float, default=0.0, help="Weight for epipolar consistency auxiliary loss (bayesian_tri model only)")
    parser.add_argument("--use_adaptive_gn", type=lambda s: s.lower() in {"1", "true"}, default=True, help="Enable adaptive Gauss-Newton refinement for bayesian_tri model (1/true or 0/false)")
    parser.add_argument("--anisotropic_covariance", type=lambda s: s.lower() in {"1", "true"}, default=True, help="Use anisotropic 2-D covariance for bayesian_tri model (1/true or 0/false)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_datasets = []
    for tp in args.train:
        train_datasets.append(RandomClipDataset(tp, args.clip_len, n_samples=args.train_samples))
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = TemporalClipDataset(args.val, args.clip_len, stride=args.val_stride)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    sample = np.load(args.train[0])
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    print(f"n_views={n_views}, j={j}, clip_len={args.clip_len}, d={args.d}, model_type={args.model_type}, "
          f"n_st_layers={args.n_st_layers}, n_view_layers={args.n_view_layers}, n_temporal_layers={args.n_temporal_layers}, "
          f"residual_hidden={args.residual_hidden}, principal_point_hidden={args.principal_point_hidden}, "
          f"principal_point_max_offset={args.principal_point_max_offset}, focal_max_scale={args.focal_max_scale}")

    if args.model_type == "dynamic_gate":
        model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointDynamicGate(
            j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
            return_pp_delta=True,
            return_raw=args.return_raw_3d or args.reproj_raw_weight > 0.0,
            return_gate=True,
        ).to(device)
    elif args.model_type == "epipolar":
        model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolar(
            j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
            return_pp_delta=True,
        ).to(device)
    elif args.model_type == "epipolar_bias_v2_pp":
        model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2(
            j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
            return_pp_delta=True,
        ).to(device)
    elif args.model_type == "epipolar_bias_v2_lite_pp":
        model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2Lite(
            j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
            return_pp_delta=True,
        ).to(device)
    elif args.model_type == "graph_skeleton_residual":
        model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraphSkeletonResidual(
            j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
            return_pp_delta=True,
        ).to(device)
    elif args.model_type == "bayesian_tri":
        model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBayesianTri(
            j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
            return_pp_delta=True,
            return_covariance=False,
            use_adaptive_gn=args.use_adaptive_gn,
            anisotropic_covariance=args.anisotropic_covariance,
        ).to(device)
    elif args.model_type == "crossview_contrast":
        model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCrossViewContrast(
            j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
            return_pp_delta=True,
        ).to(device)
    elif args.model_type == "kinematic_chain":
        model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointKinematicChain(
            j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
            return_pp_delta=True,
        ).to(device)
    elif args.model_type == "splat":
        model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSplat(
            j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
            return_pp_delta=args.pp_loss_weight > 0.0 or args.focal_max_scale > 0.0,
            return_covariance=args.splat_loss_weight > 0.0,
        ).to(device)
    elif args.model_type == "factorized":
        model = RayAttentionFusionModelTemporalCrossviewFactorizedResidualPrincipalPoint(
            j=j, d=args.d, n_views=n_views,
            n_view_layers=args.n_view_layers, n_temporal_layers=args.n_temporal_layers,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
            return_pp_delta=args.pp_loss_weight > 0.0 or args.focal_max_scale > 0.0,
            return_raw=args.return_raw_3d or args.reproj_raw_weight > 0.0,
        ).to(device)
    elif args.model_type == "camera_conditioned_pp":
        model = RayAttentionFusionModelTemporalCrossviewResidualCameraConditioned(
            j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
            return_pp_delta=args.pp_loss_weight > 0.0 or args.focal_max_scale > 0.0,
            return_raw=args.return_raw_3d or args.reproj_raw_weight > 0.0,
        ).to(device)
    elif args.model_type == "hierarchical_view_temporal_joint_pp":
        model = RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint(
            j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
            return_pp_delta=True,
            return_raw=args.return_raw_3d or args.reproj_raw_weight > 0.0,
            n_view_groups=args.n_view_groups,
            n_view_layers=args.n_view_layers,
            n_temporal_layers=args.n_temporal_layers,
            n_joint_graph_layers=args.n_joint_graph_layers,
            use_skeleton_graph=True,
        ).to(device)
    elif args.model_type == "deeper_temporal_pp":
        model = RayAttentionFusionModelHierarchicalViewDeeperTemporalResidualPrincipalPoint(
            j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
            return_pp_delta=True,
            return_raw=args.return_raw_3d or args.reproj_raw_weight > 0.0,
            n_view_groups=args.n_view_groups,
            n_view_layers=args.n_view_layers,
            n_temporal_layers=args.n_temporal_layers,
            n_joint_graph_layers=args.n_joint_graph_layers,
            use_skeleton_graph=True,
        ).to(device)
    else:
        model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
            j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers,
            residual_hidden=args.residual_hidden,
            principal_point_hidden=args.principal_point_hidden,
            principal_point_max_offset=args.principal_point_max_offset,
            focal_max_scale=args.focal_max_scale,
            return_pp_delta=args.pp_loss_weight > 0.0 or args.focal_max_scale > 0.0,
            return_raw=args.return_raw_3d or args.reproj_raw_weight > 0.0,
        ).to(device)
    gate_loss_fn = None
    if args.model_type == "dynamic_gate":
        gate_loss_fn = ViewSelectionLoss(
            sparsity_weight=args.gate_sparsity_weight,
            entropy_weight=args.gate_entropy_weight,
        )
    if args.warm_start is not None:
        state = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: missing keys when warm-starting: {missing[:5]}")
        if unexpected:
            print(f"Warning: unexpected keys when warm-starting (ignored): {unexpected[:5]}")
        print(f"Warm-started from {args.warm_start}")
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Optional pre-training phase for the principal-point correction head only.
    # This prevents the residual MLP from compensating for a spurious constant PP offset
    # and forces the PP head to actually learn the inverse perturbation.
    if args.pp_pretrain_epochs > 0:
        print(f"Pre-training PP correction head for {args.pp_pretrain_epochs} epochs...")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.principal_point_correction.parameters():
            p.requires_grad = True
        pretrain_optimizer = optim.Adam(model.principal_point_correction.parameters(), lr=args.lr)
        for pe in range(1, args.pp_pretrain_epochs + 1):
            model.train()
            train_loss = 0.0
            for xb, yb, K, R, t in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                K, R, t = K.to(device), R.to(device), t.to(device)
                xb = augment_clip(xb, view_dropout_rate=0.0, min_views=args.min_views)
                K, R, t, true_pp_delta, true_focal_scale = perturb_cameras_with_delta(
                    K, R, t,
                    rot_std=0.0,
                    trans_std=0.0,
                    focal_std=0.0,
                    pp_std=args.cam_aug_pp,
                )
                pretrain_optimizer.zero_grad()
                outputs = model(xb, K=K, R=R, t=t)
                pred = outputs[0]
                pred_pp_delta = outputs[2]
                B, T = yb.shape[:2]
                true_pp_delta = true_pp_delta.to(device).unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, -1, 2)
                loss = criterion(pred_pp_delta, -true_pp_delta)
                # Do not add reprojection loss during PP-head pre-training: the
                # rest of the model is frozen at random weights, so the
                # reprojection term would dominate and drown the PP offset
                # supervision.
                loss.backward()
                pretrain_optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)
            print(f"  PP pretrain epoch {pe}: loss={train_loss:.6f}")
        print("Unfreezing full model for end-to-end training.")
        for p in model.parameters():
            p.requires_grad = True

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        if args.cam_aug_schedule == "extrinsic_curriculum":
            ramp = min(1.0, epoch / max(1, args.cam_aug_ramp_epochs))
            schedule_rot = args.cam_aug_rot * ramp
            schedule_trans = args.cam_aug_trans * ramp
            schedule_focal = args.cam_aug_focal
            schedule_pp = args.cam_aug_pp
        elif args.cam_aug_schedule == "intrinsics_curriculum":
            ramp = min(1.0, epoch / max(1, args.cam_aug_intrinsics_ramp_epochs))
            schedule_rot = args.cam_aug_rot
            schedule_trans = args.cam_aug_trans
            schedule_focal = args.cam_aug_focal * ramp
            schedule_pp = args.cam_aug_pp * ramp
        else:
            schedule_rot = args.cam_aug_rot
            schedule_trans = args.cam_aug_trans
            schedule_focal = args.cam_aug_focal
            schedule_pp = args.cam_aug_pp
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_clip(xb, view_dropout_rate=args.view_dropout_rate, min_views=args.min_views,
                              view_noise_std=args.view_noise_std, joint_dropout_rate=args.joint_dropout_rate)
            K, R, t, true_pp_delta, true_focal_scale = perturb_cameras_with_delta(
                K, R, t,
                rot_std=schedule_rot,
                trans_std=schedule_trans,
                focal_std=schedule_focal,
                pp_std=schedule_pp,
            )
            optimizer.zero_grad()
            if args.model_type == "crossview_contrast":
                outputs = model.forward_with_contrastive_loss(xb, K=K, R=R, t=t)
            else:
                outputs = model(xb, K=K, R=R, t=t)
            pred = outputs[0]
            loss = criterion(pred, yb)
            if args.model_type == "crossview_contrast":
                loss = loss + outputs[-1]
            if args.pp_loss_weight > 0.0:
                pred_pp_delta = outputs[2]  # (B*T, V, 2)
                B, T = yb.shape[:2]
                true_pp_delta = true_pp_delta.to(device).unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, -1, 2)
                # The correction layer *adds* predicted delta to the perturbed principal point,
                # so the target is the negative of the applied offset.
                loss = loss + args.pp_loss_weight * criterion(pred_pp_delta, -true_pp_delta)
                if args.focal_max_scale > 0.0:
                    pred_focal_scale = outputs[3]  # (B*T, V)
                    true_focal_scale = true_focal_scale.to(device).squeeze(-1).unsqueeze(1).expand(B, T, -1)
                    target_focal_scale = 1.0 / true_focal_scale.reshape(B * T, -1)
                    focal_loss_weight = args.focal_loss_weight if args.focal_loss_weight is not None else args.pp_loss_weight
                    loss = loss + focal_loss_weight * criterion(pred_focal_scale, target_focal_scale)
            if args.model_type == "dynamic_gate":
                gate_weights = outputs[-2]  # (B, T, V, J) or (B*T, V, J)
                gate_reg_loss = gate_loss_fn(gate_weights)
                loss = loss + gate_reg_loss
            if args.reproj_weight > 0.0:
                points_2d = xb[..., :2]
                conf = xb[..., 2]
                loss_reproj = reprojection_loss(pred, points_2d, K, R, t, confidences=conf)
                loss = loss + args.reproj_weight * loss_reproj
            if args.reproj_raw_weight > 0.0 or args.reproj_refined_weight > 0.0:
                points_2d = xb[..., :2]
                conf = xb[..., 2]
                mask = (conf > 0) if args.reproj_mask_dropout else None
                loss_type = "charbonnier" if args.reproj_robust else "mse"
                if args.reproj_raw_weight > 0.0:
                    raw_3d = outputs[-1]
                    loss = loss + args.reproj_raw_weight * robust_reprojection_loss(
                        raw_3d, points_2d, K, R, t,
                        confidences=conf, mask=mask, loss_type=loss_type,
                    )
                if args.reproj_refined_weight > 0.0:
                    loss = loss + args.reproj_refined_weight * robust_reprojection_loss(
                        pred, points_2d, K, R, t,
                        confidences=conf, mask=mask, loss_type=loss_type,
                    )
            if args.velocity_loss_weight > 0.0:
                loss = loss + args.velocity_loss_weight * velocity_loss(pred, yb)
            if args.splat_loss_weight > 0.0:
                log_std = outputs[-1]  # (B, T, J, 3)
                points_2d = xb[..., :2]
                conf = xb[..., 2]
                loss_splat = gaussian_splatting_pose_loss(
                    pred, points_2d, K, R, t, log_std, confidences=conf,
                )
                loss = loss + args.splat_loss_weight * loss_splat
            if args.model_type == "bayesian_tri":
                epi_loss = outputs[-1]  # scalar
                loss = loss + args.epipolar_loss_weight * epi_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_err = evaluate(model, val_loader, device)
        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), output_path)
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm (saved)")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm")

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
