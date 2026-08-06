"""Train mixed-dataset temporal residual model with principal-point correction.

Extends ``train_mixed_dataset.py`` by adding camera-perturbation augmentation and
an explicit principal-point offset supervision loss.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.perturb import perturb_cameras_with_delta
from motionflow_mv.data.mixed_dataset import DATASET_IDS, DATASET_REGISTRY, build_mixed_dataloaders
from motionflow_mv.data.temporal_clip_dataset import set_seed
from motionflow_mv.fusion.ray_attention_temporal_mixed_residual_principal_point_model import RayAttentionFusionModelTemporalMixedResidualPrincipalPoint


def augment_clip(x: torch.Tensor, noise_std: float = 0.5, dropout_rate: float = 0.1) -> torch.Tensor:
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = (torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device) > dropout_rate).float()
        x[..., 2] = x[..., 2] * mask
    return x


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    per_dataset_err = {}
    per_dataset_count = {}
    for xb, yb, K, R, t, ids in loader:
        xb, yb = xb.to(device), yb.to(device)
        K, R, t = K.to(device), R.to(device), t.to(device)
        ids = ids.to(device)
        outputs = model(xb, K=K, R=R, t=t, dataset_ids=ids)
        pred, mask = outputs[0], outputs[1]
        err = (pred - yb).norm(dim=-1) * mask.float()
        total_err += err.sum().item()
        total_count += mask.sum().item()

        # Per-dataset metrics (ids are constant across the batch clip).
        for did in ids.unique().tolist():
            did_mask = (ids == did).to(device)
            if did_mask.sum() == 0:
                continue
            did_err = err[did_mask].sum().item()
            did_count = mask[did_mask].sum().item()
            per_dataset_err[did] = per_dataset_err.get(did, 0.0) + did_err
            per_dataset_count[did] = per_dataset_count.get(did, 0.0) + did_count

    per_dataset = {did: per_dataset_err[did] / per_dataset_count[did] for did in per_dataset_err}
    return total_err / total_count, per_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Train mixed-dataset temporal ray-attention fusion with principal-point correction"
    )
    parser.add_argument("--mpi_train", type=str, nargs="+", default=[], help="MPI-INF-3DHP train .npz files")
    parser.add_argument("--aist_train", type=str, nargs="+", default=[], help="AIST++ train .npz files")
    parser.add_argument("--h36m_train", type=str, nargs="+", default=[], help="Human3.6M train .npz files")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--val_dataset", type=str, required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--principal_point_hidden", type=int, default=64)
    parser.add_argument("--principal_point_max_offset", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=500)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pp_loss_weight", type=float, default=0.1)
    parser.add_argument("--focal_loss_weight", type=float, default=None, help="Weight for focal scale supervision loss (defaults to pp_loss_weight)")
    parser.add_argument("--focal_max_scale", type=float, default=0.0, help="Maximum predicted focal-length scale; 0 disables focal correction")
    parser.add_argument("--reproj_weight", type=float, default=0.0)
    parser.add_argument("--noise_std", type=float, default=0.5)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--cam_aug_rot", type=float, default=0.5)
    parser.add_argument("--cam_aug_trans", type=float, default=0.005)
    parser.add_argument("--cam_aug_focal", type=float, default=0.01)
    parser.add_argument("--cam_aug_pp", type=float, default=5.0)
    parser.add_argument("--balance_datasets", action="store_true", help="Use dataset-balanced sampling so each epoch samples equally from H36M and MPI")
    parser.add_argument("--balance_samples_per_dataset", type=int, default=None, help="Override number of samples per dataset when balance_datasets is enabled")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/ray_attention_temporal_mixed_pp.pth")
    parser.add_argument("--smoke", action="store_true", help="Override hyperparameters for a fast CPU smoke test")
    args = parser.parse_args()

    if args.smoke:
        args.d = 8
        args.n_temporal_layers = 1
        args.residual_hidden = 16
        args.principal_point_hidden = 16
        args.train_samples = 4
        args.batch_size = 2
        args.epochs = 1
        args.clip_len = 9
        args.num_workers = 0
        print("Smoke mode: overriding hyperparameters to tiny values.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_paths = {"mpi": args.mpi_train, "aist": args.aist_train, "h36m": args.h36m_train}

    train_loader, val_loader = build_mixed_dataloaders(
        train_paths=train_paths,
        val_path=args.val,
        val_dataset=args.val_dataset,
        clip_len=args.clip_len,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        val_stride=args.val_stride,
        num_workers=args.num_workers,
        balance_datasets=args.balance_datasets,
        balance_samples_per_dataset=args.balance_samples_per_dataset,
        balance_seed=args.seed,
    )

    model = RayAttentionFusionModelTemporalMixedResidualPrincipalPoint(
        d=args.d,
        n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
        principal_point_hidden=args.principal_point_hidden,
        principal_point_max_offset=args.principal_point_max_offset,
        focal_max_scale=args.focal_max_scale,
        return_pp_delta=args.pp_loss_weight > 0.0 or args.focal_max_scale > 0.0,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for xb, yb, K, R, t, ids in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            ids = ids.to(device)
            xb = augment_clip(xb, noise_std=args.noise_std, dropout_rate=args.dropout_rate)

            # Apply camera perturbation.
            K_pert, R_pert, t_pert, true_pp_delta, true_focal_scale = perturb_cameras_with_delta(
                K, R, t,
                rot_std=args.cam_aug_rot,
                trans_std=args.cam_aug_trans,
                focal_std=args.cam_aug_focal,
                pp_std=args.cam_aug_pp,
            )

            outputs = model(xb, K=K_pert, R=R_pert, t=t_pert, dataset_ids=ids)
            pred, mask = outputs[0], outputs[1]

            loss = (((pred - yb) ** 2).sum(dim=-1) * mask.float()).sum() / mask.sum()

            if args.pp_loss_weight > 0.0:
                pred_pp_delta = outputs[2]  # (B*T, V, 2)
                B, T = yb.shape[:2]
                true_pp_delta = true_pp_delta.to(device).unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, -1, 2)
                # Correction layer adds predicted delta to perturbed principal point;
                # target is the negative of the applied offset.
                loss = loss + args.pp_loss_weight * criterion(pred_pp_delta, -true_pp_delta)
                if args.focal_max_scale > 0.0:
                    pred_focal_scale = outputs[3]  # (B*T, V)
                    true_focal_scale = true_focal_scale.to(device).squeeze(-1).unsqueeze(1).expand(B, T, -1)
                    target_focal_scale = 1.0 / true_focal_scale.reshape(B * T, -1)
                    focal_loss_weight = args.focal_loss_weight if args.focal_loss_weight is not None else args.pp_loss_weight
                    loss = loss + focal_loss_weight * criterion(pred_focal_scale, target_focal_scale)

            if args.reproj_weight > 0.0:
                # TODO: implement mixed-dataset reprojection loss
                pass

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * mask.sum().item()
            train_count += mask.sum().item()

        train_loss /= train_count
        val_err, per_dataset_err = evaluate(model, val_loader, device)
        _id_to_name = {v: k for k, v in DATASET_IDS.items()}
        per_dataset_str = ", ".join(
            f"{_id_to_name.get(k, 'did_' + str(k))}={v*1000:.2f}mm"
            for k, v in per_dataset_err.items()
        )
        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), output_path)
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm (saved), {per_dataset_str}")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm, {per_dataset_str}")

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
