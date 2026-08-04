# Iter11+ Benchmark Protocol Roadmap for MotionFlow-MultiView

## Current State

We have a strong single-model line of sight: the latest `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1` fuses ray-attention, temporal/spatio-temporal attention, cross-view attention, uncertainty-weighted DLT, differentiable Gauss-Newton triangulation, and a residual MLP. The previous best MPI-INF-3DHP cross-subject MPJPE is **11.17 mm** (`ray_attention_temporal_crossview_residual` on train S1 seq1/seq2 → val S2 seq1). However, the new advanced model has **no standardized benchmark harness** yet, and the project still mixes smoke/validation/full splits across ad-hoc scripts. For ICRA/CVPR 2027, we must move from "best validation number" to a reproducible, literature-aligned benchmark protocol.

## Concrete, Implementable Improvements

### 1. Adopt a Unified `BenchmarkProtocol` Class

Create `motionflow_mv/eval/benchmark_protocol.py` that every model and dataset calls. It should encapsulate:

- Exact train/val/test splits.
- Metric computation (MPJPE, PA-MPJPE, PCK@50/100/150, AUC, per-joint/per-action/per-subject breakdowns).
- Temporal-window handling (stride, overlap, center-frame extraction) so all models report on the **same frames**.
- Unit scaling (m → mm) and root-relative evaluation where required.
- Random-seed pinning and run metadata logging.

This removes the current duplication between `experiments/eval_*.py` scripts and guarantees that any new model plugs into the same protocol.

### 2. Literature-Aligned Splits

**MPI-INF-3DHP**: Stop reporting only S2/Seq1. Train on S1–S5, validate on S2/Seq1 (early stopping only), and report the official **test set S6–S8** mean MPJPE/PA-MPJPE/PCK/AUC. Add root-relative MPJPE (pelvis index 0) because MPI-INF-3DHP literature standardizes on it.

**Human3.6M**: The current S1→S5 single-action result (5.74 mm) is encouraging but not the canonical protocol. Implement **train on S1, S5, S6, S7, S8 (all actions 2–16); test on S9 and S11 (all actions)** with per-action and mean rows. This is a known blocker: test-subject preprocessing is corrupted (S9 DLT-vs-GT ≈ 736 mm, S11 ≈ 71,402 mm); fix the camera-name grouping in `prepare_h36m_multiview.py` before any H36M numbers are claimed.

**Shelf / Campus / 3DPW / Panoptic**: Keep them as **zero-shot cross-dataset generalization** benchmarks. Report absolute MPJPE and a relative-to-DLT improvement so readers can judge transfer even when ground-truth scales differ.

### 3. Robustness and Runtime Benchmark Suite

Extend `experiments/eval_residual_robustness_mpiinf3dhp_v1.py` into a **model-agnostic** suite:

- **2D Gaussian noise**: σ ∈ {0, 2, 5, 10, 20} px.
- **Structured occlusion**: per-joint, per-view, and whole-camera dropout.
- **Outlier injection**: 2%, 5%, 10%, 20% of detections replaced by 2D outliers.
- **Camera perturbation**: small errors in K, R, t to assess calibration sensitivity.

Also add an **efficiency protocol** (building on `experiments/benchmark_inference_v3.py`) that reports parameters, FLOPs, latency at B=1 and throughput at B=32, and ONNX-export feasibility. For a 2027 paper, wall-clock cost is as important as accuracy.

### 4. Statistical Rigor

- Run each experiment with **3–5 seeds** and report mean ± std for the best model.
- Add paired t-test / Wilcoxon support in `motionflow_mv/eval/stats.py` to compare two checkpoints on the same test set.
- Log every run (git commit hash, hyperparameters, data hashes, seed) in a JSON manifest under `outputs/<run_id>/manifest.json`.

### 5. New Evaluation Metrics

- **Root-relative MPJPE** for MPI-INF-3DHP and H36M.
- **Velocity MPJPE**: mean over `‖(P_t − P_{t−1}) − (GT_t − GT_{t−1})‖` to penalize jitter, which is critical for a temporal model.
- **Bone-length error**: mean absolute difference between predicted and GT bone lengths.
- **Per-joint/per-action/per-subject tables**: necessary for paper tables and failure analysis.

### 6. A Single Eval Entry Point

Add `experiments/run_benchmark.py --model <class> --dataset <npz_or_cfg> --split test --out_dir outputs/benchmarks/<name>`. It will load a checkpoint, run the model through `BenchmarkProtocol`, and emit a `results.json` plus a paper-ready markdown table. This prevents subtle differences in stride, temporal handling, or scaling between eval scripts.

## Recommended Experiments

1. **Full MPI-INF-3DHP S6–S8 test** for the new advanced model and the 11.17 mm cross-view residual baseline. Target: mean test MPJPE ≤ 15 mm.
2. **H36M full protocol** after fixing S9/S11 preprocessing. Target: mean test MPJPE ≤ 6 mm.
3. **Ablation on the advanced model** using the same protocol: remove uncertainty, Gauss-Newton, residual, and cross-view one at a time; quantify each component’s gain.
4. **Robustness sweep** on the best checkpoint; compare against DLT baseline and the 11.17 mm residual model.
5. **Zero-shot transfer** to Shelf/Campus/3DPW/Panoptic with a standardized report template.
6. **Seed-rerun (×3)** of the best single configuration for variance reporting.

## Metrics to Track

| Metric | Why it matters |
|---|---|
| MPJPE / PA-MPJPE | Primary accuracy; must be reported in mm. |
| PCK@50/100/150, AUC | Standard MPI-INF-3DHP/H36M metrics. |
| Root-relative MPJPE | Literature-aligned MPI-INF-3DHP reporting. |
| Velocity MPJPE | Temporal consistency; differentiates temporal models. |
| Bone-length error | 3D skeleton plausibility. |
| Params / FLOPs / latency | Required for venue acceptance. |
| Robustness ΔMPJPE at noise/occlusion/outlier levels | Shows practical applicability. |

## Risks

- **Data generation**: Full MPI-INF-3DHP S4–S8 and full H36M test subjects require several GB of `.npz` outputs and GPU time. These must be gitignored.
- **H36M preprocessing bug**: S9/S11 camera grouping must be fixed before any H36M full-protocol claim.
- **Overfitting S2/Seq1**: the 11.17 mm number is on validation; moving to S6–S8 may reveal a gap. Allocate a held-out S2/Seq2 or S3 for hyperparameter search, and never use S6–S8 until final reporting.
- **Model complexity**: the advanced model has more components and may overfit with limited data. Monitor training/validation curves and use early stopping.
- **Compute**: local RTX 4090 only; schedule full runs overnight or provision cloud GPU time.

## Proposed Code Snippet

Below is a minimal `BenchmarkProtocol` skeleton that unifies evaluation. Integrating it will replace the scattered `evaluate()` functions in `experiments/eval_*.py`.

```python
# motionflow_mv/eval/benchmark_protocol.py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .metrics import compute_all_metrics


@dataclass
class BenchmarkConfig:
    dataset: str                # e.g. "mpiinf3dhp", "h36m"
    split: str                  # "train", "val", "test"
    clip_len: int = 13
    stride: int = 1
    root_joint: int = 0         # pelvis
    unit_scale: float = 1000.0  # m -> mm


class BenchmarkProtocol:
    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self.frames_evaluated: List[int] = []

    def evaluate_model(self, model, dataloader, device) -> Dict:
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb, K, R, t in dataloader:
                xb = xb.to(device)
                yb = yb.to(device)
                K, R, t = K.to(device), R.to(device), t.to(device)
                pred, *_ = model(xb, K=K, R=R, t=t)
                preds.append(pred.cpu().numpy())
                gts.append(yb.cpu().numpy())
        pred = np.concatenate(preds, axis=0)  # (N, T, J, 3)
        gt = np.concatenate(gts, axis=0)

        # Flatten temporal dimension for per-frame metrics.
        pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
        gt = gt.reshape(-1, gt.shape[-2], gt.shape[-1])
        pred_mm = pred * self.cfg.unit_scale
        gt_mm = gt * self.cfg.unit_scale

        report = compute_all_metrics(pred_mm, gt_mm)

        # Root-relative metric.
        pred_rel = pred_mm - pred_mm[:, self.cfg.root_joint:self.cfg.root_joint + 1]
        gt_rel = gt_mm - gt_mm[:, self.cfg.root_joint:self.cfg.root_joint + 1]
        report["root_rel_mpjpe"] = float(np.linalg.norm(pred_rel - gt_rel, axis=-1).mean())

        # Velocity error.
        pred_vel = np.diff(pred_mm, axis=0)
        gt_vel = np.diff(gt_mm, axis=0)
        report["velocity_mpjpe"] = float(np.linalg.norm(pred_vel - gt_vel, axis=-1).mean())

        return report

    def run(self, model, dataloader, device, out_dir: Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = self.evaluate_model(model, dataloader, device)
        with open(out_dir / "results.json", "w") as f:
            json.dump(report, f, indent=2)
        return report
```

## Summary

The path to ICRA/CVPR 2027 is not another architecture tweak—it is a **benchmarking discipline**: a single `BenchmarkProtocol`, literature-aligned train/test splits, root-relative and velocity metrics, robustness and runtime suites, and seed-replicated statistics. Implementing the class above and the six recommended experiments will transform the current best-validation number into a publishable, reproducible result set.
