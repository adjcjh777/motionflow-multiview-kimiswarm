"""CPU-only smoke test for iter-17 visibility + uncertainty v1.

Combines the visibility-gated cross-view residual model with a per-view,
per-joint log-variance head.  The DLT weight becomes

    weight = sigmoid(weight_head) * confidence * visibility * exp(-log_var)

and the training objective combines 3-D MSE, a visibility BCE loss, and the
reprojection negative-log-likelihood from the uncertainty head.

This smoke test uses the same tiny synthetic multi-view pattern as
``train_bayesian_tri_v3_smoke.py`` so it is CPU-only and needs no real data.

Usage
-----
    python experiments/prototypes/iter17_visibility_uncertainty_v1_smoke.py
    python experiments/prototypes/iter17_visibility_uncertainty_v1_smoke.py --epochs 3

Output
------
    - prints per-epoch train/val metrics
    - saves the best checkpoint to ``outputs/iter17_visibility_uncertainty_v1_smoke.pth``
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from motionflow_mv.models.crossview_residual_visibility_v2 import (
    CrossviewResidualVisibilityV2,
)
from motionflow_mv.fusion.ray_attention_model import _triangulate_weighted_dlt
from motionflow_mv.fusion.ray_attention_temporal_crossview_model import (
    _cameras_to_tensors,
)


class CrossviewResidualVisibilityUncertaintyV1(CrossviewResidualVisibilityV2):
    """Visibility + uncertainty v1 model (prototype for iter-17).

    Extends ``CrossviewResidualVisibilityV2`` by adding a per-view, per-joint
    log-variance head.  The DLT weight is gated by predicted visibility and
    weighted by predicted precision.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        n_st_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        visibility_hidden: int = 64,
        visibility_threshold: float = 0.5,
        min_visible_views: int = 2,
        uncertainty_loss_weight: float = 0.1,
        log_var_min: float = -10.0,
        log_var_max: float = 10.0,
    ):
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_st_layers=n_st_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            principal_point_hidden=principal_point_hidden,
            principal_point_max_offset=principal_point_max_offset,
            focal_max_scale=focal_max_scale,
            visibility_hidden=visibility_hidden,
            visibility_threshold=visibility_threshold,
            min_visible_views=min_visible_views,
        )
        self.uncertainty_loss_weight = uncertainty_loss_weight
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        # Per-view, per-joint log-variance from the spatio-temporal features.
        self.uncertainty_head = nn.Linear(d, 1)

    def _reprojection_nll(
        self,
        points_2d: torch.Tensor,
        pred_3d: torch.Tensor,
        proj_matrices: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian reprojection negative log-likelihood (up to constants)."""
        N, V, J, _ = points_2d.shape
        ones = torch.ones(N, J, 1, device=pred_3d.device, dtype=pred_3d.dtype)
        Xh = torch.cat([pred_3d, ones], dim=-1)  # (N, J, 4)
        p_h = torch.einsum("nvij,nkj->nvki", proj_matrices, Xh)  # (N, V, J, 3)
        z = p_h[..., 2:3].clamp(min=1e-6)
        p_proj = p_h[..., :2] / z  # (N, V, J, 2)
        err_sq = (p_proj - points_2d).pow(2).sum(dim=-1)  # (N, V, J)
        nll = 0.5 * (err_sq * torch.exp(-log_var) + log_var)
        return nll.mean()

    def forward(self, x, cameras=None, K=None, R=None, t=None):
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device

        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            K, R, t = _cameras_to_tensors(cameras, device)

        # Prepare per-sample camera tensors and flatten time into batch.
        if K.dim() == 3:
            K = K.unsqueeze(0).expand(B * T, -1, -1, -1)
            R = R.unsqueeze(0).expand(B * T, -1, -1, -1)
            t = t.unsqueeze(0).expand(B * T, -1, -1)
        elif K.dim() == 4:
            K = K.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            R = R.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            t = t.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, V, 3)
        else:
            raise ValueError("K must have shape (V, 3, 3) or (B, V, 3, 3)")

        x_flat = x.reshape(B * T, V, J, 3)
        points_2d = x_flat[..., :2]
        confidences = x_flat[..., 2]

        # Principal-point / intrinsic correction before ray embedding.
        correction_outputs = self.principal_point_correction(
            K=K,
            x=x_flat,
            weights=confidences,
        )
        K_corrected = correction_outputs[0]

        # Per-frame v3 features (uses corrected intrinsics).
        feat = self._extract_frame_features(x_flat, K_corrected, R, t)  # (B*T, V, J, d)

        # Spatio-temporal (time + view) attention.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        for layer in self.st_transformer:
            feat = layer(feat)
        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        # Visibility gating (soft multiplier in [0, 1]).
        visibility = self._visibility_multiplier(feat, confidences)  # (B*T, V, J)

        # Per-view, per-joint log-variance prediction.
        feat_for_uncertainty = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        log_var = self.uncertainty_head(feat_for_uncertainty).squeeze(-1)  # (B*T, J, V)
        log_var = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)
        log_var = log_var.permute(0, 2, 1)  # (B*T, V, J)

        # Variance-weighted DLT: lower variance -> higher precision weight.
        precision = torch.exp(-log_var)

        # Per-frame weight prediction and triangulation with corrected intrinsics.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences * visibility * precision  # (B*T, V, J)
        weights = weights.clamp(min=1e-4)

        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K_corrected @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        # Residual refinement head.
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        residual_input = torch.cat([feat_pooled, pred_3d_raw], dim=-1)  # (B*T, J, d+3)
        delta = self.residual_mlp(residual_input)  # (B*T, J, 3)
        pred_3d = pred_3d_raw + delta

        # Auxiliary reprojection NLL so uncertainties are supervised.
        nll_loss = self._reprojection_nll(points_2d, pred_3d, P, log_var)
        nll_loss = self.uncertainty_loss_weight * nll_loss

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)
        visibility = visibility.view(B, T, V, J)
        log_var = log_var.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            visibility = visibility.squeeze(1)
            log_var = log_var.squeeze(1)

        return pred_3d, weights, visibility, log_var, nll_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_synthetic_cameras(n_views: int = 4):
    """Build a circular rig of pinhole cameras looking at the origin.

    The camera z-axis points from the camera center toward the origin so that
    the subject is in front of every camera (positive camera-z).  This keeps
    the reprojection NLL finite during the smoke test.
    """
    K_list, R_list, t_list = [], [], []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([4.0 * np.cos(theta), 4.0 * np.sin(theta), 1.5])
        forward = -c / np.linalg.norm(c)  # camera z-axis (into the scene)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, forward], axis=0)
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


def _project_points(joints_3d: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Project world points into all views."""
    X = joints_3d  # (F, J, 3)
    t = t[:, None, None, :]  # (V, 1, 1, 3)
    X_cam = torch.einsum("vab,fjb->vfja", R, X) + t  # (V, F, J, 3)
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv = torch.matmul(K[:, None, None], (X_cam / z)[..., None])  # (V, F, J, 3, 1)
    points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
    return points_2d.permute(1, 0, 2, 3)  # (F, V, J, 2)


class SyntheticVisibilityUncertaintyDataset(torch.utils.data.Dataset):
    """Tiny synthetic multi-view pose dataset for smoke tests.

    Optionally drops one random view per clip so the visibility head receives
    a non-trivial supervision signal.
    """

    def __init__(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        n_frames: int = 100,
        n_joints: int = 17,
        clip_len: int = 9,
        noise_std: float = 0.5,
        view_dropout_prob: float = 0.2,
    ):
        self.K = K
        self.R = R
        self.t = t
        self.n_joints = n_joints
        self.clip_len = clip_len
        self.noise_std = noise_std
        self.view_dropout_prob = view_dropout_prob

        torch.manual_seed(42)
        joints_3d = torch.randn(n_frames, n_joints, 3) * 0.3  # (F, J, 3)
        for _ in range(2):
            joints_3d[1:-1] = 0.5 * joints_3d[1:-1] + 0.25 * (joints_3d[:-2] + joints_3d[2:])

        points_2d = _project_points(joints_3d, K, R, t)
        if noise_std > 0:
            points_2d = points_2d + torch.randn_like(points_2d) * noise_std

        self.points_2d = points_2d  # (F, V, J, 2)
        self.confidences = torch.ones_like(points_2d[..., 0])  # (F, V, J)
        self.joints_3d = joints_3d  # (F, J, 3)

        self.total_frames = n_frames
        self.num_clips = max(1, self.total_frames - clip_len + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx
        end = start + self.clip_len
        conf = self.confidences[start:end].clone()
        if self.view_dropout_prob > 0:
            V = conf.shape[1]
            for _ in range(self.clip_len):
                if random.random() < self.view_dropout_prob:
                    view = random.randint(0, V - 1)
                    conf[:, view, :] = 0.0
        x = torch.cat([self.points_2d[start:end], conf.unsqueeze(-1)], dim=-1)
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def evaluate(model, loader, device, criterion, visibility_loss_weight, uncertainty_loss_weight):
    model.eval()
    total_loss = 0.0
    total_err = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, weights, visibility, log_var, nll_loss = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)
            if visibility_loss_weight > 0.0:
                visible_target = (xb[..., 2] > 0).float().to(device)
                loss = loss + visibility_loss_weight * F.binary_cross_entropy(visibility, visible_target)
            loss = loss + nll_loss
            total_loss += loss.item() * xb.size(0)
            total_err += (pred - yb).norm(dim=-1).mean().item() * xb.size(0)
            count += xb.size(0)
    return total_loss / count, total_err / count


def main():
    parser = argparse.ArgumentParser(description="CPU smoke test for iter-17 visibility + uncertainty v1")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--clip_len", type=int, default=9, help="Temporal clip length")
    parser.add_argument("--d", type=int, default=32, help="Feature dimension")
    parser.add_argument("--n_st_layers", type=int, default=1, help="Spatio-temporal transformer layers")
    parser.add_argument("--residual_hidden", type=int, default=64, help="Residual MLP hidden size")
    parser.add_argument("--principal_point_hidden", type=int, default=32, help="Principal-point head hidden size")
    parser.add_argument("--visibility_hidden", type=int, default=32, help="Visibility head hidden size")
    parser.add_argument("--visibility_loss_weight", type=float, default=0.1, help="Weight for visibility BCE loss")
    parser.add_argument("--uncertainty_loss_weight", type=float, default=0.1, help="Weight for uncertainty NLL loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu only for smoke)")
    parser.add_argument("--output", type=str, default="outputs/iter17_visibility_uncertainty_v1_smoke.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else "cpu")
    print(f"Device: {device}")

    K, R, t = _make_synthetic_cameras(n_views=4)
    n_joints = 17
    n_train_frames = 80
    n_val_frames = 30

    train_dataset = SyntheticVisibilityUncertaintyDataset(
        K, R, t,
        n_frames=n_train_frames,
        n_joints=n_joints,
        clip_len=args.clip_len,
        view_dropout_prob=0.2,
    )
    val_dataset = SyntheticVisibilityUncertaintyDataset(
        K, R, t,
        n_frames=n_val_frames,
        n_joints=n_joints,
        clip_len=args.clip_len,
        view_dropout_prob=0.0,
    )

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

    model = CrossviewResidualVisibilityUncertaintyV1(
        j=n_joints,
        d=args.d,
        n_views=4,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        principal_point_hidden=args.principal_point_hidden,
        principal_point_max_offset=20.0,
        focal_max_scale=0.0,
        visibility_hidden=args.visibility_hidden,
        visibility_threshold=0.5,
        min_visible_views=2,
        uncertainty_loss_weight=args.uncertainty_loss_weight,
        log_var_min=-10.0,
        log_var_max=10.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"n_views=4, j={n_joints}, clip_len={args.clip_len}, d={args.d}, "
        f"n_st_layers={args.n_st_layers}, residual_hidden={args.residual_hidden}, "
        f"params={n_params}"
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, Kb, Rb, tb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            Kb, Rb, tb = Kb.to(device), Rb.to(device), tb.to(device)
            optimizer.zero_grad()
            pred, weights, visibility, log_var, nll_loss = model(xb, K=Kb, R=Rb, t=tb)
            loss = criterion(pred, yb)
            if args.visibility_loss_weight > 0.0:
                visible_target = (xb[..., 2] > 0).float().to(device)
                loss = loss + args.visibility_loss_weight * F.binary_cross_entropy(visibility, visible_target)
            loss = loss + nll_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_loss, val_err = evaluate(model, val_loader, device, criterion, args.visibility_loss_weight, args.uncertainty_loss_weight)
        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), output_path)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"val_MPJPE={val_err*1000:.2f}mm (saved)"
            )
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"val_MPJPE={val_err*1000:.2f}mm"
            )

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
