# Iter11+ MotionFlow Integration Report

**Date:** 2026-08-04  
**Topic:** `motionflow_integration` — bringing the all-in-one ray-attention fusion model into the end-to-end MotionFlow multi-view pipeline.  
**Target venue:** ICRA/CVPR 2027.

## 1. Current state

The codebase contains a new advanced fusion model, `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1` (`motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`), that stacks every developed component on top of the spatio-temporal cross-view transformer:

- ray-aware per-view and joint-level attention;
- spatio-temporal `(time, view)` self-attention;
- uncertainty-weighted DLT;
- differentiable Gauss–Newton triangulation;
- residual MLP refinement.

A training script exists at `experiments/train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.py`, but it trains on one or two MPI-INF-3DHP sequences with `RandomClipDataset`.

The current best published result is the cross-view residual model at **11.17 mm** MPJPE on MPI-INF-3DHP validation (S1→S2/Seq1). The combined model has **no published eval**, no `FusionModule` wrapper, no registration in `FUSION_REGISTRY`, and no mixed-dataset loader. In short, the model exists as a standalone class but is not yet a first-class citizen of the MotionFlow multi-view pipeline.

## 2. Concrete, implementable improvements

### A. Plug the model into the `FusionModule` registry

All existing pipeline consumers (`MultiViewAdapter`, demo scripts, the `MultiViewPipeline`) look up backends through `FUSION_REGISTRY`. The combined model must expose a `FusionModule` that:

- consumes `(T, V, J, 2)` keypoints and `(T, V, J)` confidences;
- instantiates the model and loads a checkpoint;
- returns `(T, J, 3)` world-coordinate joints.

The model’s forward returns a 4-tuple `(pred_3d, weights, log_var, nll_loss)`; the wrapper should discard the extra outputs for the registry contract but optionally expose them via `uncertainty` metadata.

### B. Unify the training data loader

The current script only accepts MPI-INF-3DHP `.npz` files. H36M WebBridge conversion is in progress at `data/webbridge/h36m`. We need a single `MultiviewClipDataset` that can:

- read any WebBridge `.npz` whose keys are `points_2d`, `confidences`, `joints_3d`, `camera_K/R/t`;
- optionally map different skeleton definitions to a common joint set (MPI uses 28 joints, H36M uses 17/32);
- mix MPI and H36M clips in one training run and report per-dataset validation curves.

### C. Add a dedicated evaluation harness

There is no evaluation script for the combined model. A new script, e.g. `experiments/eval_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1.py`, should:

- load a trained checkpoint;
- run the full validation sequence with temporal windows;
- compute MPJPE, PA-MPJPE, PCK@50/100/150 mm, PCK-AUC (0–150 mm);
- report per-joint and per-view breakdowns;
- report mean reprojection error (pixels) and the predicted per-view uncertainty;
- support robustness perturbations (Gaussian noise, view dropout, calibration noise).

### D. Fix the loss composition

The training script currently adds an external `reprojection_loss` **and** the model’s internal `nll_loss`, which is a form of double reprojection supervision. The integration should:

- keep only the model-internal `nll_loss` (or make the external one optional);
- add a bone-length/skeleton-consistency auxiliary loss to exploit human skeleton structure;
- use learning-rate decay and gradient clipping to stabilize the multi-component objective.

### E. Enable test-time refinement and checkpoint warm-start

- Allow `n_iter > 1` in the residual head at inference.
- Support warm-start by loading the base cross-view residual checkpoint with `strict=False` into the combined model.
- Add horizontal-flip test-time augmentation across symmetric views and average predictions.

### F. Log paper-quality diagnostics

For interpretability and the camera-ready paper, log:

- histograms of per-view predicted weights/uncertainties;
- scatter plots of predicted uncertainty vs. actual reprojection error;
- residual correction magnitude per joint;
- robustness curves vs. noise/dropout level.

## 3. Recommended experiments

| # | Experiment | What to vary | Success metric |
|---|------------|--------------|----------------|
| 1 | Full MPI-INF-3DHP combined model | Train on all MPI train subjects/sequences, warm-start from the 11.17 mm cross-view residual checkpoint | MPJPE on S2/Seq1 |
| 2 | Component ablation | Remove uncertainty head, GN head, or cross-view attention one at a time | ΔMPJPE vs. full model |
| 3 | Mixed MPI + H36M training | Use the unified loader with both datasets; freeze vs. fine-tune | MPI val MPJPE, H36M S11/S5 val MPJPE |
| 4 | Robustness benchmark | Add 0–5 px noise, 0–50 % view dropout, ±5°/±5 cm calibration jitter | MPJPE degradation curve |
| 5 | Longer temporal context | `clip_len = 27, 49` with the cross-view transformer | Memory/time, MPJPE |

Metrics to track for every run:
- MPJPE, PA-MPJPE, PCK@50/100/150 mm, PCK-AUC(0–150 mm)
- per-joint MPJPE, per-view MPJPE
- mean reprojection error (px)
- wall-clock training time and inference fps
- model parameter count

## 4. Risks and mitigations

- **Compute and memory.** The cross-view transformer attends over `T·V` tokens per joint. With `T=49` and `V=14`, memory grows quickly. Mitigation: keep `n_st_layers=2` and use gradient checkpointing, smaller batch size, or shorter clips first.
- **Loss conflicts.** Uncertainty NLL and residual refinement can pull the estimate in different directions. Mitigation: start with the cross-view residual warm checkpoint and freeze the base transformer for the first few epochs.
- **Data format mismatch.** MPI uses 28 joints, H36M 17/32. Mitigation: train a 17-joint common skeleton or keep dataset-specific output heads.
- **Silent checkpoint loading failures.** `load_state_dict(..., strict=False)` may leave new heads randomly initialized. Mitigation: print loaded/missing keys explicitly and verify uncertainty-head gradients.
- **Double reprojection loss.** As noted, the external `reprojection_loss` and internal `nll_loss` overlap. Mitigation: make the external loss optional and default it to `0.0`.

## 5. Proposed code: `FusionModule` wrapper

Create `motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_module.py` and register it in `motionflow_mv/fusion/__init__.py`:

```python
from typing import List
import numpy as np
import torch

from ..calibration.camera import Camera
from .fusion_module import FusionModule, FUSION_REGISTRY
from .ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model import (
    RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1,
)


class RayAttentionTemporalCrossviewUncertaintyResidualLearnedTriV1FusionModule(FusionModule):
    name = "ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1"

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        checkpoint_path: str | None = None,
        n_iter: int = 1,
        input_scale: float = 1.0,
    ):
        super().__init__()
        self.input_scale = input_scale
        self.n_iter = n_iter
        self.model = RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1(
            j=j, d=d, n_views=n_views
        )
        if checkpoint_path is not None:
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu", weights_only=True),
                strict=False,
            )
        self.model.eval()

    def fuse(
        self,
        points_2d: np.ndarray,
        confidences: np.ndarray,
        cameras: List[Camera],
    ) -> np.ndarray:
        points_2d = np.asarray(points_2d, dtype=np.float32)
        confidences = np.asarray(confidences, dtype=np.float32)
        if points_2d.ndim == 3:
            points_2d = points_2d[None]
            confidences = confidences[None]

        x = np.concatenate([points_2d, confidences[..., None]], axis=-1)
        x_tensor = torch.from_numpy(x).to(next(self.model.parameters()).device)

        with torch.no_grad():
            pred, *_ = self.model(x_tensor, cameras=cameras, n_iter=self.n_iter)
        return pred.cpu().numpy()


def register_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_fusion_module():
    FUSION_REGISTRY.register(RayAttentionTemporalCrossviewUncertaintyResidualLearnedTriV1FusionModule())
```

## 6. Next actions

1. Add the wrapper above and register it.
2. Refactor `experiments/train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.py` to use a unified multi-dataset loader, optional external reprojection loss, and explicit warm-start loading.
3. Write the evaluation/robustness script for the combined model.
4. Run the component-ablation sweep (Experiment 2) on the full MPI-INF-3DHP validation set to identify which additions actually improve the 11.17 mm baseline.
5. Once H36M conversion completes, run the mixed-dataset training (Experiment 3) and report cross-dataset MPJPE.
