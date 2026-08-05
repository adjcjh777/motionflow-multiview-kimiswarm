"""CPU-friendly occlusion robustness evaluation / smoke test.

Usage (smoke, no checkpoint needed)
-----------------------------------
    python experiments/eval_occlusion_robustness.py

Usage (real evaluation)
-----------------------
    python experiments/eval_occlusion_robustness.py \
        --checkpoint outputs/ray_attention_temporal_crossview_residual_pp.pth \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --out_dir outputs/occlusion_robustness

The script is intentionally lightweight and runs on CPU by default; it does not
start any training.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.occlusion_aug import (
    OcclusionAugmenter,
    occlude_joints,
    occlude_views,
    random_occlude_joints,
    random_occlude_views,
)
from motionflow_mv.eval.metrics import compute_all_metrics
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class TemporalClipDataset(torch.utils.data.Dataset):
    """Yield clips of shape (T, V, J, 3) from a canonical .npz file."""

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
        self.num_clips = max(1, (self.total_frames - clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )  # (T, V, J, 3)
        y = self.joints_3d[start:end]  # (T, J, 3)
        return x, y, self.K, self.R, self.t


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def apply_occlusion(x, mode, rate, seed=42):
    """Apply one occlusion condition to an input tensor.

    Args:
        x: Tensor of shape (..., V, J, C).
        mode: One of ``"view"``, ``"joint"``, ``"view_joint"``, or ``"clean"``.
        rate: Occlusion rate in [0, 1].
        seed: Reproducibility seed for the random generator.

    Returns:
        Augmented tensor (x is not modified in-place).
    """
    if mode == "clean" or rate <= 0.0:
        return x

    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)

    if mode == "view":
        return random_occlude_views(x, rate, generator=generator)
    if mode == "joint":
        return random_occlude_joints(x, rate, per_view=True, generator=generator)
    if mode == "view_joint":
        x = random_occlude_views(x, rate, generator=generator)
        x = random_occlude_joints(x, rate * 0.5, per_view=True, generator=generator)
        return x
    raise ValueError(f"Unknown occlusion mode: {mode}")


def build_model(j, n_views, args):
    return RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=j,
        d=args.d,
        n_views=n_views,
        n_heads=args.n_heads,
        n_joint_layers=args.n_joint_layers,
        n_st_layers=args.n_st_layers,
        max_temporal_len=args.max_temporal_len,
        residual_hidden=args.residual_hidden,
        principal_point_hidden=args.principal_point_hidden,
        principal_point_max_offset=args.principal_point_max_offset,
    )


def evaluate(model, loader, device, occlusion_mode, occlusion_rate, seed=42):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb = apply_occlusion(xb, occlusion_mode, occlusion_rate, seed=seed)
            xb = xb.to(device)
            yb = yb.to(device)
            K = K.to(device)
            R = R.to(device)
            t = t.to(device)
            pred, *_ = model(xb, K=K, R=R, t=t)
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())

    preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], 3) * 1000.0
    gts = np.concatenate(gts, axis=0).reshape(-1, gts[0].shape[-2], 3) * 1000.0
    return compute_all_metrics(preds, gts)


def smoke_test():
    """Lightweight CPU smoke test for the occlusion augmentation API."""
    print("=" * 60)
    print("Occlusion augmentation CPU smoke test")
    print("=" * 60)

    torch.manual_seed(0)
    B, T, V, J, C = 2, 3, 4, 17, 3
    x = torch.rand(B, T, V, J, C)
    x[..., 2] = 1.0  # full confidence before occlusion

    # 1. Explicit view occlusion
    x_view = occlude_views(x.clone(), view_indices=[0, 2])
    assert x_view.shape == x.shape
    occluded_view = x_view[0, 0, [0, 2], :, 2].sum().item() == 0.0
    assert occluded_view, "Explicit view occlusion did not zero confidence"

    # 2. Explicit joint occlusion
    x_joint = occlude_joints(x.clone(), joint_indices=[5, 10])
    assert x_joint[0, 0, :, [5, 10], 2].sum().item() == 0.0, "Explicit joint occlusion failed"

    # 3. Random view occlusion
    x_rv = random_occlude_views(x, rate=0.5, generator=torch.Generator().manual_seed(7))
    assert x_rv[..., 2].eq(0).any(), "Random view occlusion did not drop any view"

    # 4. Random joint occlusion
    x_rj = random_occlude_joints(
        x, rate=0.3, per_view=True, generator=torch.Generator().manual_seed(8)
    )
    assert x_rj[..., 2].eq(0).any(), "Random joint occlusion did not drop any joint"

    # 5. Augmenter wrapper
    augmenter = OcclusionAugmenter(view_rate=0.2, joint_rate=0.1, seed=123)
    x_aug = augmenter(x)
    assert x_aug.shape == x.shape, "Augmenter changed tensor shape"
    assert x_aug[..., 2].eq(0).any(), "Augmenter did not occlude anything"

    # 6. Model forward with occlusion (random weights, CPU only)
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=J,
        d=16,
        n_views=V,
        n_heads=2,
        n_joint_layers=1,
        n_st_layers=1,
        max_temporal_len=64,
        residual_hidden=32,
    )
    model.eval()
    K = torch.eye(3).unsqueeze(0).repeat(V, 1, 1)
    R = torch.eye(3).unsqueeze(0).repeat(V, 1, 1)
    t = torch.zeros(V, 3)
    with torch.no_grad():
        pred, *_ = model(x_aug, K=K, R=R, t=t)
    assert pred.shape[-2:] == (J, 3), f"Model output shape mismatch: {pred.shape}"

    print("All smoke checks passed.")
    print(f"  Input shape      : {list(x.shape)}")
    print(f"  View-occluded    : {(x_rv[..., 2] == 0).sum().item()} / {x_rv[..., 2].numel()} detections")
    print(f"  Joint-occluded   : {(x_rj[..., 2] == 0).sum().item()} / {x_rj[..., 2].numel()} detections")
    print(f"  Augmenter-occluded: {(x_aug[..., 2] == 0).sum().item()} / {x_aug[..., 2].numel()} detections")
    print(f"  Model output     : {list(pred.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Occlusion robustness evaluation (CPU smoke by default)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_joint_layers", type=int, default=1)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--max_temporal_len", type=int, default=256)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--principal_point_hidden", type=int, default=64)
    parser.add_argument("--principal_point_max_offset", type=float, default=20.0)
    parser.add_argument("--out_dir", type=str, default="outputs/occlusion_robustness")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.checkpoint is None or args.dataset is None:
        smoke_test()
        return

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = TemporalClipDataset(args.dataset, args.clip_len, stride=args.stride)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    data = np.load(args.dataset)
    n_views = int(data["camera_K"].shape[0])
    j = int(data["points_2d"].shape[2])

    model = build_model(j, n_views, args).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset} | views={n_views}, joints={j}, clips={len(dataset)}")

    conditions = [
        {"name": "clean", "mode": "clean", "rate": 0.0},
        {"name": "view_drop_20", "mode": "view", "rate": 0.2},
        {"name": "view_drop_40", "mode": "view", "rate": 0.4},
        {"name": "joint_drop_20", "mode": "joint", "rate": 0.2},
        {"name": "joint_drop_40", "mode": "joint", "rate": 0.4},
        {"name": "view_joint_20_10", "mode": "view_joint", "rate": 0.2},
    ]

    results = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "n_views": n_views,
        "n_joints": j,
        "clip_len": args.clip_len,
        "conditions": [],
    }

    print("\n=== Occlusion robustness evaluation ===")
    for cond in conditions:
        report = evaluate(model, loader, device, cond["mode"], cond["rate"], seed=args.seed)
        entry = {
            "name": cond["name"],
            "mode": cond["mode"],
            "rate": cond["rate"],
            "mpjpe_mm": float(report["mpjpe"]),
            "pa_mpjpe_mm": float(report["pa_mpjpe"]),
        }
        results["conditions"].append(entry)
        print(
            f"  {cond['name']:20s} | MPJPE {entry['mpjpe_mm']:6.2f} mm | "
            f"PA-MPJPE {entry['pa_mpjpe_mm']:6.2f} mm"
        )

    out_json = out_dir / "occlusion_robustness.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_json}")


if __name__ == "__main__":
    main()
