"""Reusable robustness evaluation matrix for the temporal-residual model family.

Loads a trained checkpoint by ``model_type``, evaluates a validation ``.npz``
under a fixed set of input/calibration perturbations, and writes both a JSON
report and a Markdown summary table.

Perturbation axes
-----------------
* clean baseline
* view dropout (0.1, 0.3, 0.5)
* 2-D keypoint Gaussian noise (0.5, 1.0, 2.0 px std)
* principal-point perturbation (5.0, 10.0 px std)

Usage
-----
    # CPU by default
    python experiments/run_robustness_matrix.py \
        --model epipolar_bias_v2_pp \
        --checkpoint outputs/epipolar_bias_v2_smoke.pth \
        --dataset tmp/mpi_s02_seq01_smoke.npz \
        --output_dir outputs/robustness_matrix_epipolar_bias_v2_pp

    # CUDA explicitly
    python experiments/run_robustness_matrix.py \
        --model epipolar_bias_v2_pp \
        --checkpoint outputs/epipolar_bias_v2_smoke.pth \
        --dataset tmp/mpi_s02_seq01_smoke.npz \
        --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.occlusion_aug import random_occlude_views
from motionflow_mv.eval.metrics import compute_all_metrics


# Reuse model registry, dataset, collate, and builder from eval_full_metrics.py.
_EFM_PATH = Path(__file__).with_name("eval_full_metrics.py")
spec = importlib.util.spec_from_file_location("eval_full_metrics", _EFM_PATH)
eval_full_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_full_metrics)

MODEL_CLASSES = eval_full_metrics.MODEL_CLASSES
build_model = eval_full_metrics.build_model
TemporalClipDataset = eval_full_metrics.TemporalClipDataset
collate_fn = eval_full_metrics.collate_fn


# Conditions evaluated by this matrix.
CONDITIONS = [
    {"name": "clean"},
    {"name": "view_dropout_0.1", "view_dropout": 0.1},
    {"name": "view_dropout_0.3", "view_dropout": 0.3},
    {"name": "view_dropout_0.5", "view_dropout": 0.5},
    {"name": "noise_std_0.5", "noise_std": 0.5},
    {"name": "noise_std_1.0", "noise_std": 1.0},
    {"name": "noise_std_2.0", "noise_std": 2.0},
    {"name": "cam_aug_pp_5.0", "cam_aug_pp": 5.0},
    {"name": "cam_aug_pp_10.0", "cam_aug_pp": 10.0},
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_view_dropout(x: torch.Tensor, rate: float, generator: torch.Generator) -> torch.Tensor:
    """Randomly drop whole views across the batch."""
    return random_occlude_views(x, rate, per_sample=False, generator=generator)


def apply_noise(x: torch.Tensor, noise_std: float, generator: torch.Generator) -> torch.Tensor:
    """Add pixel-domain Gaussian noise to 2-D keypoint coordinates."""
    if noise_std <= 0.0:
        return x
    noise = torch.randn(x[..., :2].shape, generator=generator, device=x.device) * noise_std
    x = x.clone()
    x[..., :2] = x[..., :2] + noise
    return x


def apply_camera_perturbation(
    K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, pp_std: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perturb camera intrinsics (principal point only)."""
    if pp_std <= 0.0:
        return K, R, t
    from motionflow_mv.calibration.perturb import perturb_intrinsics_with_delta
    K_aug, _, _ = perturb_intrinsics_with_delta(K, focal_std=0.0, pp_std=pp_std)
    return K_aug, R, t


def evaluate_condition(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    condition: dict,
    seed: int,
) -> dict:
    """Evaluate one robustness condition and return metric summary."""
    model.eval()
    generator = torch.Generator(device=device.type if device.type in {"cuda", "cpu"} else "cpu")
    generator.manual_seed(seed)

    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            K = K.to(device)
            R = R.to(device)
            t = t.to(device)

            if condition.get("view_dropout", 0.0) > 0.0:
                xb = apply_view_dropout(xb, condition["view_dropout"], generator)

            if condition.get("noise_std", 0.0) > 0.0:
                xb = apply_noise(xb, condition["noise_std"], generator)

            if condition.get("cam_aug_pp", 0.0) > 0.0:
                K, R, t = apply_camera_perturbation(K, R, t, condition["cam_aug_pp"])

            out = model(xb, K=K, R=R, t=t)
            pred = out[0] if isinstance(out, tuple) else out
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # (N, T, J, 3)
    gts = np.concatenate(gts, axis=0)
    preds = preds.reshape(-1, preds.shape[-2], 3) * 1000.0  # meters -> mm
    gts = gts.reshape(-1, gts.shape[-2], 3) * 1000.0

    report = compute_all_metrics(preds, gts)
    summary = {
        k: float(v)
        for k, v in report.items()
        if not k.endswith("_per_joint") and not isinstance(v, np.ndarray)
    }
    return summary


def build_markdown(results: dict) -> str:
    lines = [
        "| Condition | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | PCK@100 | PCK@150 | AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['mpjpe']:.2f} | {r['pa_mpjpe']:.2f} | "
            f"{r['pck@50mm']:.3f} | {r['pck@100mm']:.3f} | {r['pck@150mm']:.3f} | {r['pck_auc']:.3f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run a fixed robustness evaluation matrix for a trained model."
    )
    parser.add_argument("--model", type=str, choices=list(MODEL_CLASSES), required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for JSON/Markdown outputs. Defaults to outputs/robustness_matrix_<model_type>/",
    )
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--n_view_layers", type=int, default=2)
    parser.add_argument("--n_view_groups", type=int, default=2)
    parser.add_argument("--n_joint_graph_layers", type=int, default=1)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--graph_layers", type=int, default=3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--target_k", type=int, default=4)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--gt_scale", type=float, default=1.0)
    parser.add_argument("--camera_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on. Defaults to 'cpu' for CPU-only evaluation.",
    )
    parser.add_argument(
        "--source_n_views",
        type=int,
        default=None,
        help="Source view count of the trained model; enables variable-view inference when target differs.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.output_dir is None:
        output_dir = Path("outputs") / f"robustness_matrix_{args.model}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset)
    n_views = int(data["camera_K"].shape[0])
    j = int(data["points_2d"].shape[2])
    print(f"Dataset: {args.dataset} | views={n_views} | joints={j}")

    dataset = TemporalClipDataset(
        args.dataset,
        args.clip_len,
        stride=args.val_stride,
        gt_scale=args.gt_scale,
        camera_scale=args.camera_scale,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    source_n_views = args.source_n_views if args.source_n_views is not None else n_views
    model = build_model(args, source_n_views, j).to(device)
    missing, unexpected = model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True),
        strict=False,
    )
    if missing:
        print(f"Warning: missing keys in checkpoint: {missing[:5]}")
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint (ignored): {unexpected[:5]}")

    results = {}
    print(f"{'Condition':<20} {'MPJPE':>10} {'PA-MPJPE':>10} {'PCK@50':>8} {'PCK@100':>8} {'PCK@150':>8} {'AUC':>8}")
    print("-" * 90)
    for condition in CONDITIONS:
        summary = evaluate_condition(model, loader, device, condition, args.seed)
        results[condition["name"]] = summary
        print(
            f"{condition['name']:<20} {summary['mpjpe']:>10.2f} {summary['pa_mpjpe']:>10.2f} "
            f"{summary['pck@50mm']:>8.3f} {summary['pck@100mm']:>8.3f} "
            f"{summary['pck@150mm']:>8.3f} {summary['pck_auc']:>8.3f}"
        )

    json_path = output_dir / "robustness_matrix.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "model": args.model,
                "dataset": str(args.dataset),
                "device": str(device),
                "seed": args.seed,
                "conditions": results,
            },
            f,
            indent=2,
        )
    print(f"Saved JSON to {json_path}")

    md = build_markdown(results)
    md_path = output_dir / "robustness_matrix.md"
    md_path.write_text(md)
    print(f"Saved Markdown to {md_path}")


if __name__ == "__main__":
    main()
