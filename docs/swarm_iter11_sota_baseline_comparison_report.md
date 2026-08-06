# Iter11+ Research Report: SOTA Baseline Comparison

## 1. Current State

The MotionFlow-MultiView project has produced a family of calibrated multi-view fusion models in `motionflow_mv/fusion/`. The current best results on the MPI-INF-3DHP cross-subject benchmark (S1 → S2/Seq1) are:

| Model | Params | MPJPE (mm) | PA-MPJPE (mm) | AUC |
|---|---:|---:|---:|---:|
| Raw confidence-weighted DLT | — | 25.21 | — | — |
| Temporal ray-attention (no residual) | 218 k | 25.21 | 24.14 | 0.832 |
| Temporal-residual (5 epochs, d=64, h=128) | 243 k | **11.17** | **8.24** | 0.926 |
| Temporal-residual small (d=32, h=64) | 66 k | 13.22 | 11.77 | 0.912 |
| Cross-view spatio-temporal residual | 1.06 M | 15.29 | 13.49 | 0.898 |
| Fast temporal-residual + reprojection | — | 47.54 | — | — |

The 11.17 mm checkpoint (`outputs/ray_attention_temporal_residual_final5.pth`) is the strongest validated model. The 47.54 mm result came from a limited-data/epoch run and is not representative. A new combined model was just added: `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1` (`motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`), which fuses cross-view spatio-temporal attention, uncertainty-weighted DLT, differentiable Gauss-Newton triangulation, and a residual MLP. It has not yet been trained to convergence.

On Human3.6M, the temporal-residual model reaches 5.74 mm MPJPE and 3.99 mm PA-MPJPE on the S1→S5 cross-subject task.

## 2. Gap Analysis for SOTA Baseline Comparison

The biggest weakness for an ICRA/CVPR 2027 submission is not the model, but the **baseline comparison table**. The current paper draft (`docs/paper_draft_icra_cvpr_2027.md`) mixes real results with placeholder or incorrect citations:

- “Iskandar, et al. Triangulation learning. CVPR, 2020” — the actual paper is **Iskakov et al., “Learnable Triangulation of Human Pose,” ICCV 2019**.
- “Ray-attention multi-view pose. CVPR, 2022” — no such canonical paper exists; it appears to reference the project’s own earlier work.
- The related work section lists VoxelPose, EpipolarPose, MeTRAbs, RPSNet/MvMESH, but none are run or compared numerically.
- There is no comparison with recent SOTA such as MV-SSM (CVPR 2025), RUMPL (arXiv 2025), or COMPOSE (arXiv 2026), despite the literature-gap report identifying them as directly relevant.

**A submission with invented citations and missing SOTA numbers will be rejected on desk review.** The immediate priority is to replace the related-work table with a **reproducible, numerically grounded SOTA comparison**.

## 3. Concrete, Implementable Improvements

### 3.1 Implement a canonical “Learnable Triangulation” reproduction

The closest direct competitor is **Iskakov et al., ICCV 2019**, which predicts per-view weights and triangulates with weighted DLT. Our temporal-residual model already subsumes this architecture. We should expose a stripped-down variant that exactly matches the Iskakov recipe:

- Per-view CNN (or MLP) feature extractor on 2D keypoints + confidence.
- View-wise attention or average pooling to produce per-view weights.
- Weighted DLT triangulation.
- No temporal transformer, no residual head.

This gives a clean ablation: `DLT → Iskakov-style → temporal → residual → cross-view+uncertainty+GN`.

### 3.2 Add a lightweight EpipolarPose-style self-supervised baseline

EpipolarPose (Kocabas et al., CVPR 2019) shows that epipolar geometry can train a 3D pose estimator without 3D GT. We can implement a self-supervised baseline using our own 2D keypoints:

- Train the temporal-residual model with only the **reprojection loss** and an multi-view epipolar consistency loss on the MPI-INF-3DHP 2D projections.
- Use the 3D GT only for evaluation, not training.

This directly tests whether our strong 3D-supervised numbers require 3D labels, which is a strong paper claim.

### 3.3 Add a volumetric baseline (VoxelPose-style)

VoxelPose operates in 3D voxel space and is heavy, but a lightweight proxy can be built from the existing pipeline:

- Back-project each view’s 2D heatmap/confidence along its ray.
- Accumulate a sparse 3D confidence volume per joint.
- Take the soft-argmax or maximum-likelihood 3D location.

Use this as an upper-bound baseline. If our ray-attention model is within a few mm of this much heavier method while being 100× smaller, it strengthens the paper’s “lightweight” claim.

### 3.4 Implement per-method robustness curves

`experiments/baselines.py` already provides DLT, RANSAC-DLT, robust Huber, and temporal smoothing. We should extend it to report, on the same validation split:

- Clean MPJPE
- With 5 px / 20 px Gaussian 2D noise
- With 10% / 20% / 50% random joint occlusion
- With 5% / 10% 2D outlier replacement
- With 1-view / 2-view dropout

Run the same corruption protocol for every baseline and our best model. The current draft reports only a single robustness table; a full curve by perturbation level is expected by reviewers.

### 3.5 Standardize cross-dataset generalization benchmark

The literature-gap report (`docs/swarm_iter5/literature_gap.md`) identified cross-calibration generalization as a key differentiator. We should run:

- **MPI-INF-3DHP → Human3.6M**: train on MPI S1, evaluate on H36M S5 without fine-tuning.
- **Human3.6M → MPI-INF-3DHP**: train on H36M S1, evaluate on MPI S2/Seq1.
- **Shelf → Campus / Campus → Shelf**: zero-shot cross-dataset transfer using the plugin contract.

Report MPJPE, PA-MPJPE, and reprojection error. This turns a potential weakness (small per-dataset size) into a strength.

### 3.6 Add SOTA methods from 2025–2026

The literature-gap report lists MV-SSM, RUMPL, COMPOSE, and DisPOSE. We cannot reproduce all of them, but we can:

- For **MV-SSM** and **RUMPL**, cite their reported MPI-INF-3DHP / H36M numbers in a comparison table and note where our method differs (camera-conditioned ray embeddings vs. state-space scan / generic ray transformer).
- For **COMPOSE**, implement a minimal optimization-based baseline (multi-view 2D-to-3D via epipolar/graph cover) if code is available; otherwise cite.
- For **MeTRAbs**, note it is monocular and not a direct multi-view competitor, but include its single-view H36M number as a reference point.

## 4. Proposed Code Change: Unified SOTA Comparison Script

Create `experiments/compare_sota_baselines.py` that evaluates every relevant baseline and our best model on the same validation split with the same corruption protocol. Below is the core skeleton.

```python
# experiments/compare_sota_baselines.py
import json
from pathlib import Path
import numpy as np
import torch

from motionflow_mv.eval.metrics import mpjpe, pa_mpjpe, pck, pck_auc
from motionflow_mv.fusion.triangulation import triangulate_dlt_torch
from motionflow_mv.fusion.ray_attention_temporal_residual_module import (
    RayAttentionTemporalResidualFusionModule,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model import (
    RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1,
)


def evaluate_model(name, model, loader, device, corrupt_fn=None):
    preds, gts = [], []
    model.eval()
    with torch.no_grad():
        for x, y, K, R, t in loader:
            x = x.to(device)
            if corrupt_fn is not None:
                x = corrupt_fn(x)
            pred = model(x, K=K.to(device), R=R.to(device), t=t.to(device))[0]
            preds.append(pred.cpu().numpy().reshape(-1, pred.shape[-2], 3))
            gts.append(y.numpy().reshape(-1, y.shape[-2], 3))
    pred = np.concatenate(preds)
    gt = np.concatenate(gts)
    return {
        "MPJPE": mpjpe(pred, gt) * 1000,
        "PA-MPJPE": pa_mpjpe(pred, gt) * 1000,
        "PCK@50mm": pck(pred, gt, 0.05),
        "PCK@150mm": pck(pred, gt, 0.15),
        "AUC": pck_auc(pred, gt, max_threshold=0.15)[0],
    }


def dlt_baseline(loader):
    """Confidence-weighted DLT using triangulate_dlt_torch."""
    preds, gts = [], []
    for x, y, K, R, t in loader:
        # x: (B, T, V, J, 3), K/R/t per clip
        pts = x[..., :2].numpy()
        conf = x[..., 2].numpy()
        # Flatten batch/time and triangulate
        pred = triangulate_dlt_torch(
            torch.from_numpy(pts).view(-1, *pts.shape[-3:]).cuda(),
            torch.from_numpy(conf).view(-1, *conf.shape[-2:]).cuda(),
            K.cuda(), R.cuda(), t.cuda(),
        )[0]
        preds.append(pred.cpu().numpy())
        gts.append(y.numpy().reshape(-1, y.shape[-2], 3))
    pred = np.concatenate(preds)
    gt = np.concatenate(gts)
    return {"MPJPE": mpjpe(pred, gt) * 1000, "PA-MPJPE": pa_mpjpe(pred, gt) * 1000}


def main():
    # 1. Load MPI-INF-3DHP or H36M validation clips.
    # 2. For each baseline/model, run evaluate_model.
    # 3. For each corruption, run evaluate_model with corrupt_fn.
    # 4. Write JSON table to outputs/sota_comparison.json.
    pass


if __name__ == "__main__":
    main()
```

The script should be run as:

```bash
conda run -n mf python experiments/compare_sota_baselines.py \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --checkpoint outputs/ray_attention_temporal_residual_final5.pth \
    --output outputs/sota_comparison_mpi.json
```

## 5. Experiments to Run

1. **Reproduce Iskakov-style baseline.** Train a model with only view-level attention + weighted DLT, no temporal or residual components, on the same MPI-INF-3DHP split.
2. **Train the combined model to convergence.** `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1` needs at least 5 epochs on the full MPI-INF-3DHP training split. Compare against the 11.17 mm temporal-residual model.
3. **Robustness sweep.** Run `experiments/baselines.py` and the new `compare_sota_baselines.py` under noise, occlusion, outliers, and view dropout.
4. **Cross-dataset transfer.** Train on MPI S1, evaluate on H36M S5; train on H36M S1, evaluate on MPI S2/Seq1. Use the plugin contract to keep units consistent.
5. **Shelf/Campus zero-shot.** Re-run `eval_cross_dataset_generalization.py` with DLT, temporal-residual, and the combined model.
6. **Latency benchmark.** Use `experiments/benchmark_inference_v3.py` to report clips/s and ms/clip for DLT, temporal-residual, cross-view residual, and the combined model at batch sizes 1, 4, 8, 16.
7. **GVHMR/ScoreHMR real-world demo.** Project single-view SMPL output to multiple virtual cameras and compare fused DLT vs. fused temporal-residual vs. the combined model.

## 6. Metrics to Track

- **MPJPE / PA-MPJPE** in mm on MPI-INF-3DHP and Human3.6M.
- **PCK@50/100/150 mm** and **AUC**.
- **Per-joint/per-body-part breakdown** (head, torso, arms, legs) to locate failure modes.
- **Reprojection error** in pixels to confirm geometric consistency.
- **Robustness delta**: MPJPE increase under each corruption level.
- **Cross-dataset Δ**: MPJPE on target dataset vs. in-distribution validation.
- **Runtime**: ms/clip and clips/s on RTX 4090.
- **Model size**: parameter count for each method.

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Combined model overfits or converges slower than temporal-residual | Medium | High | Start with the 11.17 mm checkpoint as initialization; freeze the encoder and train only the uncertainty/GN/residual heads for the first 2 epochs. |
| Cross-view residual continues to underperform temporal-only | Medium | High | Diagnose using per-joint errors; if cross-view attention overfits, add epipolar/ray-angle bias or reduce capacity. |
| SOTA methods have no public code or numbers | Medium | Medium | Cite reported numbers honestly; clearly label “reported” vs. “our reproduction.” |
| Shelf/Campus unit/calibration mismatch contaminates numbers | Medium | High | Run the calibration-validation helper before any cross-dataset comparison; drop sequences with inconsistent camera/3D alignment. |
| GVHMR/ScoreHMR integration is brittle | Low | Medium | Scope the demo to the same synthetic virtual-camera setup first, then move to real video. |
| Paper draft citations are incorrect | High | High | Replace all placeholder references with verified BibTeX entries before any submission. |

## 8. Immediate Action Items

1. Fix citations in `docs/paper_draft_icra_cvpr_2027.md`: Iskakov et al. ICCV 2019, Tu et al. ECCV 2020, Kocabas et al. CVPR 2019, Sárándi et al. IEEE T-BIOM 2021, Chharia et al. CVPR 2025, and arXiv papers from the literature-gap report.
2. Write and run `experiments/compare_sota_baselines.py`.
3. Add a reproducible “Iskakov-style” baseline model to `motionflow_mv/fusion/`.
4. Train the combined uncertainty+GN+residual model for 5 epochs and evaluate.
5. Generate the full robustness curve and cross-dataset table for the paper.

---

*Report prepared for Iter11+ of the ICRA/CVPR 2027 submission roadmap, focusing on sota_baseline_comparison.*
