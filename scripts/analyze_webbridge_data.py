"""Analyze the WebBridge dataset used by a given split YAML.

Computes clip counts, lengths, camera/view distributions, action label
distribution, missing-keypoint ratios, and 2D/3D pose statistics.
Saves a JSON report to ``outputs/webbridge_analysis_report.json``.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


# Human3.6M action ids -> names (from the standard H36M action set)
H36M_ACTION_NAMES = {
    1: "Directions",
    2: "Discussion",
    3: "Eating",
    4: "Greeting",
    5: "Phoning",
    6: "Photo",
    7: "Posing",
    8: "Purchases",
    9: "Sitting",
    10: "SittingDown",
    11: "Smoking",
    12: "Waiting",
    13: "WalkDog",
    14: "Walking",
    15: "WalkTogether",
    16: "Discussion2",
}


def parse_h36m_action(path: Path) -> str:
    """Extract H36M action id and name from a filename like s_01_acts_02_multiview_m.npz."""
    match = re.search(r"acts_(\d+)", path.name)
    if match:
        act_id = int(match.group(1))
        return f"{act_id:02d}_{H36M_ACTION_NAMES.get(act_id, 'Unknown')}"
    return "unknown"


def parse_mpi_label(path: Path) -> str:
    """Extract MPI subject/sequence label from a filename like s_01_seq_01_v14_multiview_m.npz."""
    match = re.search(r"(s_\d+_seq_\d+)", path.name)
    if match:
        return match.group(1)
    return "unknown"


def parse_label(path: Path, dataset: str) -> str:
    """Infer an action/sequence label from filename and dataset source."""
    if dataset == "h36m":
        return parse_h36m_action(path)
    elif dataset == "mpi":
        return parse_mpi_label(path)
    return "unknown"


def running_stats():
    """Return objects to compute mean, std, min, max in one pass."""
    return {
        "sum": 0.0,
        "sum_sq": 0.0,
        "count": 0,
        "min": float("inf"),
        "max": float("-inf"),
    }


def update_running(stats, value):
    """Update running mean/std/min/max with a new numpy array."""
    arr = value.reshape(-1)
    n = arr.size
    if n == 0:
        return
    stats["sum"] += float(arr.sum())
    stats["sum_sq"] += float((arr ** 2).sum())
    stats["count"] += n
    min_a = float(arr.min())
    max_a = float(arr.max())
    if min_a < stats["min"]:
        stats["min"] = min_a
    if max_a > stats["max"]:
        stats["max"] = max_a


def finalize_stats(stats):
    count = stats["count"]
    if count == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "count": 0}
    mean = stats["sum"] / count
    variance = stats["sum_sq"] / count - mean * mean
    variance = max(variance, 0.0)
    return {
        "mean": float(mean),
        "std": float(np.sqrt(variance)),
        "min": float(stats["min"]) if stats["min"] != float("inf") else None,
        "max": float(stats["max"]) if stats["max"] != float("-inf") else None,
        "count": int(count),
    }


def analyze_split(paths, names, project_root):
    clip_lengths = []
    view_counts = Counter()
    action_counter = Counter()
    dataset_counter = Counter()
    missing_ratios = []
    missing_per_view = []
    missing_per_dataset_view = []

    stats_2d = running_stats()
    stats_3d = running_stats()

    for p, dataset in zip(paths, names):
        path = Path(project_root) / p
        if not path.exists():
            print(f"[WARN] File not found: {path}")
            continue
        try:
            data = np.load(path)
            points_2d = data["points_2d"]      # (T, V, J, 2)
            confidences = data["confidences"]  # (T, V, J)
            joints_3d = data["joints_3d"]      # (T, J, 3)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            continue

        T, V, J, _ = points_2d.shape
        clip_lengths.append(T)
        view_counts[V] += 1
        dataset_counter[dataset] += 1
        action_counter[parse_label(path, dataset)] += 1

        # Missing keypoints: confidence == 0 OR all-zero 2D coordinates
        missing = (confidences == 0) | (np.all(points_2d == 0, axis=-1))
        total_kpts = missing.size
        missing_count = int(missing.sum())

        if total_kpts:
            ratio = missing_count / total_kpts
            missing_ratios.append(ratio)

        # Per-view missing ratios for this clip
        if V > 0:
            per_view = missing.reshape(T, V, J).mean(axis=(0, 2))
            missing_per_view.extend(per_view.tolist())
            missing_per_dataset_view.append({
                "dataset": dataset,
                "views": V,
                "missing_ratio_per_view": per_view.tolist(),
            })

        update_running(stats_2d, points_2d)
        update_running(stats_3d, joints_3d)

    out = {
        "num_clips": len(clip_lengths),
        "clip_length": {
            "mean": float(np.mean(clip_lengths)) if clip_lengths else 0.0,
            "std": float(np.std(clip_lengths)) if clip_lengths else 0.0,
            "min": int(np.min(clip_lengths)) if clip_lengths else 0,
            "max": int(np.max(clip_lengths)) if clip_lengths else 0,
            "total_frames": int(sum(clip_lengths)),
        },
        "camera_distribution": {
            "num_views_per_clip": {str(k): v for k, v in sorted(view_counts.items())},
        },
        "dataset_distribution": dict(dataset_counter),
        "action_label_distribution": dict(action_counter.most_common()),
        "missing_keypoints": {
            "overall_ratio": float(np.mean(missing_ratios)) if missing_ratios else 0.0,
            "per_view_ratio_mean": float(np.mean(missing_per_view)) if missing_per_view else 0.0,
            "per_clip_per_view": missing_per_dataset_view,
        },
        "pose_statistics_2d": finalize_stats(stats_2d),
        "pose_statistics_3d": finalize_stats(stats_3d),
    }

    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze WebBridge dataset split.")
    parser.add_argument(
        "--config",
        default="configs/deprecated/circular/splits/webbridge_h36m_mpi_mixed_train_val.yaml",
        help="Path to the YAML split config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root relative to which dataset paths are resolved.",
    )
    parser.add_argument(
        "--output",
        default="outputs/webbridge_analysis_report.json",
        help="Where to save the JSON report.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_paths = cfg.get("train_paths", [])
    val_paths = cfg.get("val_paths", [])
    train_names = cfg.get("train_names", [])
    val_names = cfg.get("val_names", [])

    if len(train_paths) != len(train_names):
        train_names = ["unknown"] * len(train_paths)
    if len(val_paths) != len(val_names):
        val_names = ["unknown"] * len(val_paths)

    print(f"Analyzing {len(train_paths)} train clips and {len(val_paths)} val clips...")

    train_report = analyze_split(train_paths, train_names, args.project_root)
    val_report = analyze_split(val_paths, val_names, args.project_root)

    report = {
        "config": args.config,
        "summary": {
            "total_train_clips": train_report["num_clips"],
            "total_val_clips": val_report["num_clips"],
            "total_clips": train_report["num_clips"] + val_report["num_clips"],
            "total_train_frames": train_report["clip_length"]["total_frames"],
            "total_val_frames": val_report["clip_length"]["total_frames"],
        },
        "train": train_report,
        "val": val_report,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {output_path}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
