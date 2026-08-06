"""Extended robustness evaluation matrix across noise, occlusion, and view dropout.

Extends ``experiments/run_robustness_matrix.py`` with:

* per-joint occlusion (joint dropout)
* combined perturbations (noise + occlusion + view dropout)
* a CPU-only smoke path that runs without a checkpoint or dataset

Usage (real evaluation)
-----------------------
    python experiments/prototypes/run_extended_robustness_matrix.py \
        --model bayesian_tri_v2_pp \
        --checkpoint outputs/bayesian_tri_v2_smoke.pth \
        --dataset tmp/mpi_s02_seq01_smoke.npz \
        --output_dir outputs/extended_robustness_matrix

Usage (CPU smoke test)
----------------------
    python experiments/prototypes/run_extended_robustness_matrix.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from motionflow_mv.data.occlusion_aug import random_occlude_joints, random_occlude_views
from motionflow_mv.eval.metrics import compute_all_metrics


# Reuse model registry / dataset / builder from experiments/eval_full_metrics.py.
_EFM_PATH = Path(__file__).parent.parent / "eval_full_metrics.py"
spec = importlib.util.spec_from_file_location("eval_full_metrics", _EFM_PATH)
eval_full_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_full_metrics)

MODEL_CLASSES = eval_full_metrics.MODEL_CLASSES
build_model = eval_full_metrics.build_model
TemporalClipDataset = eval_full_metrics.TemporalClipDataset
collate_fn = eval_full_metrics.collate_fn


# --------------------------------------------------------------------------- #
# Condition definitions
# --------------------------------------------------------------------------- #

NOISE_LEVELS = [0.5, 1.0, 2.0]
JOINT_OCCLUSION_RATES = [0.1, 0.2, 0.3]
VIEW_DROPOUT_RATES = [0.1, 0.3, 0.5]


def build_conditions() -> list[dict]:
    """Return the extended matrix of single-axis and combined conditions."""
    conditions = [{"name": "clean"}]

    # Single-axis noise
    for level in NOISE_LEVELS:
        conditions.append({"name": f"noise_{level:.1f}px", "noise_std": level})

    # Single-axis joint occlusion
    for rate in JOINT_OCCLUSION_RATES:
        conditions.append({"name": f"joint_occlusion_{int(rate * 100):02d}", "joint_occlusion": rate})

    # Single-axis view dropout
    for rate in VIEW_DROPOUT_RATES:
        conditions.append({"name": f"view_dropout_{int(rate * 100):02d}", "view_dropout": rate})

    # Two-axis combinations
    conditions.append({"name": "noise_1.0px_joint_occlusion_20", "noise_std": 1.0, "joint_occlusion": 0.2})
    conditions.append({"name": "noise_1.0px_view_dropout_30", "noise_std": 1.0, "view_dropout": 0.3})
    conditions.append({"name": "joint_occlusion_20_view_dropout_30", "joint_occlusion": 0.2, "view_dropout": 0.3})

    # Three-axis combination
    conditions.append({
        "name": "noise_1.0px_joint_occlusion_20_view_dropout_30",
        "noise_std": 1.0,
        "joint_occlusion": 0.2,
        "view_dropout": 0.3,
    })

    return conditions


# --------------------------------------------------------------------------- #
# Perturbation helpers
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_view_dropout(x: torch.Tensor, rate: float, generator: torch.Generator) -> torch.Tensor:
    """Randomly drop whole camera views."""
    if rate <= 0.0:
        return x
    return random_occlude_views(x, rate, per_sample=False, generator=generator)


def apply_joint_occlusion(x: torch.Tensor, rate: float, generator: torch.Generator) -> torch.Tensor:
    """Randomly zero confidences for individual (view, joint) detections."""
    if rate <= 0.0:
        return x
    return random_occlude_joints(x, rate, per_view=True, per_sample=False, generator=generator)


def apply_noise(x: torch.Tensor, noise_std: float, generator: torch.Generator) -> torch.Tensor:
    """Add pixel-domain Gaussian noise to 2-D keypoint coordinates."""
    if noise_std <= 0.0:
        return x
    x = x.clone()
    noise = torch.randn(x[..., :2].shape, generator=generator, device=x.device) * noise_std
    x[..., :2] = x[..., :2] + noise
    return x


def _condition_seed(base_seed: int, name: str) -> int:
    """Derive a deterministic but condition-unique seed."""
    h = zlib.crc32(name.encode()) & 0xFFFFFFFF
    return base_seed + h


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_condition(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    condition: dict,
    base_seed: int,
) -> dict:
    """Evaluate one robustness condition and return a metric summary."""
    model.eval()

    seed = _condition_seed(base_seed, condition["name"])
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

            xb = apply_noise(xb, condition.get("noise_std", 0.0), generator)
            xb = apply_joint_occlusion(xb, condition.get("joint_occlusion", 0.0), generator)
            xb = apply_view_dropout(xb, condition.get("view_dropout", 0.0), generator)

            out = model(xb, K=K, R=R, t=t)
            pred = out[0] if isinstance(out, tuple) else out
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    preds = preds.reshape(-1, preds.shape[-2], 3) * 1000.0  # m -> mm
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


def build_csv(results: dict) -> str:
    lines = ["condition,mpjpe_mm,pa_mpjpe_mm,pck@50mm,pck@100mm,pck@150mm,pck_auc"]
    for name, r in results.items():
        lines.append(
            f"{name},{r['mpjpe']:.4f},{r['pa_mpjpe']:.4f},"
            f"{r['pck@50mm']:.4f},{r['pck@100mm']:.4f},{r['pck@150mm']:.4f},{r['pck_auc']:.4f}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

def _make_synthetic_dataset(path: Path, n_frames: int = 30, n_views: int = 4, n_joints: int = 17) -> None:
    """Write a tiny synthetic multiview sequence to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)

    points_2d = np.random.randn(n_frames, n_views, n_joints, 2).astype(np.float32) * 100.0
    confidences = np.ones((n_frames, n_views, n_joints), dtype=np.float32)
    joints_3d = np.random.randn(n_frames, n_joints, 3).astype(np.float32) * 0.5

    K = np.zeros((n_views, 3, 3), dtype=np.float32)
    K[:, 0, 0] = 800.0
    K[:, 1, 1] = 800.0
    K[:, 2, 2] = 1.0
    K[:, :2, 2] = 320.0, 240.0

    R = np.tile(np.eye(3, dtype=np.float32)[None], (n_views, 1, 1))
    t = np.zeros((n_views, 3), dtype=np.float32)

    np.savez(
        path,
        points_2d=points_2d,
        confidences=confidences,
        joints_3d=joints_3d,
        camera_K=K,
        camera_R=R,
        camera_t=t,
    )


def smoke_test() -> dict:
    """CPU-only smoke test that exercises every condition end-to-end."""
    import argparse as _argparse

    print("=" * 60)
    print("Extended robustness matrix CPU smoke test")
    print("=" * 60)

    tmp_dir = Path(tempfile.gettempdir()) / "mf_extended_robustness_smoke"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = tmp_dir / "smoke.npz"
    checkpoint_path = tmp_dir / "smoke.pth"

    n_views, n_joints = 4, 17
    _make_synthetic_dataset(dataset_path, n_frames=30, n_views=n_views, n_joints=n_joints)

    args = _argparse.Namespace(
        model="residual",
        checkpoint=str(checkpoint_path),
        dataset=str(dataset_path),
        output_dir=str(tmp_dir / "outputs"),
        clip_len=5,
        d=16,
        n_temporal_layers=1,
        n_st_layers=1,
        n_view_layers=1,
        n_view_groups=1,
        n_joint_graph_layers=1,
        residual_hidden=8,
        graph_layers=2,
        k=4,
        target_k=4,
        min_views=2,
        batch_size=2,
        val_stride=1,
        gt_scale=1.0,
        camera_scale=1.0,
        source_n_views=None,
        seed=42,
        device="cpu",
        parents=None,
        symmetry_pairs=None,
        no_skeleton_graph=False,
    )

    device = torch.device("cpu")
    data = np.load(args.dataset)
    source_n_views = args.source_n_views if args.source_n_views is not None else n_views
    model = build_model(args, source_n_views, n_joints)
    torch.save(model.state_dict(), args.checkpoint)
    model.to(device)

    results = run(args)
    print("Smoke test passed: all conditions evaluated and outputs written.")
    print(f"  Output dir: {args.output_dir}")
    return results


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    conditions = build_conditions()
    results = {}
    print(f"{'Condition':<45} {'MPJPE':>10} {'PA-MPJPE':>10} {'PCK@50':>8} {'PCK@100':>8} {'PCK@150':>8} {'AUC':>8}")
    print("-" * 110)
    for condition in conditions:
        summary = evaluate_condition(model, loader, device, condition, args.seed)
        results[condition["name"]] = summary
        print(
            f"{condition['name']:<45} {summary['mpjpe']:>10.2f} {summary['pa_mpjpe']:>10.2f} "
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

    csv = build_csv(results)
    csv_path = output_dir / "robustness_matrix.csv"
    csv_path.write_text(csv)
    print(f"Saved CSV to {csv_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run an noise / occlusion / view-dropout robustness matrix for a trained model."
    )
    parser.add_argument("--model", type=str, choices=list(MODEL_CLASSES), default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for JSON/Markdown/CSV outputs. Defaults to outputs/extended_robustness_matrix_<model>/",
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
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--source_n_views", type=int, default=None)
    parser.add_argument("--parents", type=str, default=None)
    parser.add_argument("--symmetry_pairs", type=str, default=None)
    parser.add_argument("--no_skeleton_graph", action="store_true")
    args = parser.parse_args()

    if args.checkpoint is None or args.dataset is None or args.model is None:
        smoke_test()
        return

    if args.output_dir is None:
        args.output_dir = f"outputs/extended_robustness_matrix_{args.model}"

    run(args)


if __name__ == "__main__":
    main()
