"""Distil a lightweight real-time student from the Bayesian triangulation teacher."""
import argparse, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent.parent))
from motionflow_mv.calibration.perturb import perturb_cameras_with_delta
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBayesianTri
from motionflow_mv.models.distilled_student_principal_point_model import DistilledStudentPrincipalPointModel

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TemporalClipDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, clip_len, stride=1):
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
    def __len__(self): return self.num_clips
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat([self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)], dim=-1)
        return x, self.joints_3d[start:end], self.K, self.R, self.t

class RandomClipDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, clip_len, n_samples=2000):
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
    def __len__(self): return self.n_samples
    def __getitem__(self, idx):
        start = random.randint(0, max(0, self.total_frames - self.clip_len))
        end = start + self.clip_len
        x = torch.cat([self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)], dim=-1)
        return x, self.joints_3d[start:end], self.K, self.R, self.t

def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t

def augment_clip(x, noise_std=0.5, dropout_rate=0.1, outlier_rate=0.02, outlier_scale=100.0):
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = (torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device) > dropout_rate).float()
        x[..., 2] = x[..., 2] * mask
    if outlier_rate > 0:
        outlier_mask = torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device) < outlier_rate
        outlier = (torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], 2, device=x.device) - 0.5) * 2 * outlier_scale
        x[..., :2] = torch.where(outlier_mask[..., None], outlier, x[..., :2])
    return x

def evaluate(model, loader, device):
    model.eval()
    total_err, total_count = 0.0, 0
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
    parser = argparse.ArgumentParser(description="Distil a lightweight real-time student from the Bayesian triangulation teacher")
    parser.add_argument("--train", type=str, nargs="+", required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--teacher", type=str, default="outputs/bayesian_tri_pp_full_mpiinf3dhp.pth")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--n_st_layers", type=int, default=1)
    parser.add_argument("--residual_hidden", type=int, default=64)
    parser.add_argument("--distill_alpha", type=float, default=0.5)
    parser.add_argument("--weight_align_beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--pp_loss_weight", type=float, default=0.2)
    parser.add_argument("--cam_aug_pp", type=float, default=5.0)
    parser.add_argument("--cam_aug_focal", type=float, default=0.01)
    parser.add_argument("--cam_aug_schedule", type=str, default="intrinsics_curriculum", choices=["flat", "extrinsics_curriculum", "intrinsics_curriculum"])
    parser.add_argument("--cam_aug_intrinsics_ramp_epochs", type=int, default=5)
    parser.add_argument("--pp_pretrain_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/distilled_student_pp_mpiinf3dhp.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_datasets = [RandomClipDataset(tp, args.clip_len, n_samples=args.train_samples) for tp in args.train]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = TemporalClipDataset(args.val, args.clip_len, stride=args.val_stride)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    sample = np.load(args.train[0])
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    print(f"n_views={n_views}, j={j}, clip_len={args.clip_len}, d={args.d}, "
          f"n_st_layers={args.n_st_layers}, residual_hidden={args.residual_hidden}, "
          f"distill_alpha={args.distill_alpha}, weight_align_beta={args.weight_align_beta}")

    teacher = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBayesianTri(
        j=j, d=64, n_views=n_views, n_st_layers=2, residual_hidden=128,
        principal_point_hidden=64, return_pp_delta=True, return_covariance=False,
    ).to(device)
    state = torch.load(args.teacher, map_location=device, weights_only=True)
    missing, unexpected = teacher.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: missing keys when loading teacher: {missing[:5]}")
    if unexpected:
        print(f"Warning: unexpected keys when loading teacher: {unexpected[:5]}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher loaded from {args.teacher} ({sum(p.numel() for p in teacher.parameters()):,} params)")

    student = DistilledStudentPrincipalPointModel(
        j=j, d=args.d, n_views=n_views, n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden, principal_point_hidden=64,
        return_pp_delta=True,
    ).to(device)
    print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    if args.pp_pretrain_epochs > 0:
        print(f"Pre-training student PP correction head for {args.pp_pretrain_epochs} epochs...")
        for p in student.parameters():
            p.requires_grad = False
        for p in student.principal_point_correction.parameters():
            p.requires_grad = True
        pretrain_optimizer = torch.optim.Adam(student.principal_point_correction.parameters(), lr=args.lr)
        for pe in range(1, args.pp_pretrain_epochs + 1):
            student.train()
            train_loss = 0.0
            for xb, yb, K, R, t in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                K, R, t = K.to(device), R.to(device), t.to(device)
                xb = augment_clip(xb)
                K, R, t, true_pp_delta, true_focal_scale = perturb_cameras_with_delta(
                    K, R, t, rot_std=0.0, trans_std=0.0, focal_std=0.0, pp_std=args.cam_aug_pp,
                )
                pretrain_optimizer.zero_grad()
                outputs = student(xb, K=K, R=R, t=t)
                pred_pp_delta = outputs[2]
                B, T = yb.shape[:2]
                true_pp_delta = true_pp_delta.to(device).unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, -1, 2)
                loss = criterion(pred_pp_delta, -true_pp_delta)
                loss.backward()
                pretrain_optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)
            print(f"  PP pretrain epoch {pe}: loss={train_loss:.6f}")
        for p in student.parameters():
            p.requires_grad = True
        print("Unfreezing full student for end-to-end distillation.")

    for epoch in range(1, args.epochs + 1):
        student.train()
        train_loss = 0.0
        if args.cam_aug_schedule == "intrinsics_curriculum":
            ramp = min(1.0, epoch / max(1, args.cam_aug_intrinsics_ramp_epochs))
            schedule_focal = args.cam_aug_focal * ramp
            schedule_pp = args.cam_aug_pp * ramp
        else:
            schedule_focal = args.cam_aug_focal
            schedule_pp = args.cam_aug_pp

        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_clip(xb)
            K, R, t, true_pp_delta, true_focal_scale = perturb_cameras_with_delta(
                K, R, t, rot_std=0.5, trans_std=0.005, focal_std=schedule_focal, pp_std=schedule_pp,
            )

            with torch.no_grad():
                t_out = teacher(xb, K=K, R=R, t=t)
                teacher_pred = t_out[0]
                teacher_weights = t_out[1]

            s_out = student(xb, K=K, R=R, t=t)
            student_pred = s_out[0]
            student_weights = s_out[1]

            loss = criterion(student_pred, yb)

            if args.distill_alpha > 0.0:
                distill_loss = criterion(student_pred, teacher_pred)
                loss = (1.0 - args.distill_alpha) * loss + args.distill_alpha * distill_loss

            if args.weight_align_beta > 0.0:
                s_w = student_weights.reshape(-1, n_views)
                t_w = teacher_weights.reshape(-1, n_views)
                mask = (s_w.sum(dim=-1, keepdim=True) > 0) & (t_w.sum(dim=-1, keepdim=True) > 0)
                if mask.any():
                    s_w = s_w[mask.squeeze(-1)]
                    t_w = t_w[mask.squeeze(-1)]
                    cos_sim = F.cosine_similarity(s_w, t_w, dim=-1).mean()
                    loss = loss + args.weight_align_beta * (1.0 - cos_sim)

            if args.pp_loss_weight > 0.0:
                pred_pp_delta = s_out[2]
                B, T = yb.shape[:2]
                true_pp_delta = true_pp_delta.to(device).unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, -1, 2)
                loss = loss + args.pp_loss_weight * criterion(pred_pp_delta, -true_pp_delta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)
        val_err = evaluate(student, val_loader, device)
        if val_err < best_val:
            best_val = val_err
            torch.save(student.state_dict(), output_path)
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm (saved)")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm")

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
