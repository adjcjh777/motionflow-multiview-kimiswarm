# Iter11+ GVHMR Demo Extension Report

## 1. Current State

The project just merged the most capable fusion model to date:
`motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`.
It stacks cross-view spatio-temporal attention, per-view log-variance uncertainty,
differentiable Gauss-Newton triangulation, and a residual MLP head.
A training script exists for MPI-INF-3DHP, but the model is **not yet plugged
into the demo pipeline** and no checkpoint has been produced.

The existing GVHMR demos use older back-ends:

- `experiments/demo_gvhmr_multiview_projection.py` — `RayAttentionFusionModelV3` / v1 plugin.
- `experiments/demo_gvhmr_multiview_projection_residual.py` — `RayAttentionTemporalResidualFusionModule`.
- `experiments/demo_ir_from_gvhmr.py` — only converts `hmr4d_results.pt` to `HumanMotionIR`.

The only real GVHMR artifact, `data/gvhmr_demo/hmr4d_results.pt`, is a **single-view**
result.  All multi-view experiments therefore simulate a camera rig by projecting
the single-view world SMPL joints through virtual calibrated cameras and adding noise.
The best published MPI-INF-3DHP validation MPJPE is ~11.17 mm (cross-view residual),
so the new combined model should be pushed to beat that on a benchmark *and* be
demonstrated on the GVHMR proxy.

## 2. Concrete, Implementable Improvements

### 2.1 Plugin wrapper for the combined model

The combined model cannot be used by the pipeline because it lacks a
`FusionModule` wrapper and is not registered in `motionflow_mv/fusion/__init__.py`.
Add `RayAttentionTemporalCrossviewUncertaintyResidualLearnedTriFusionModule`.

### 2.2 Evaluation harness for the combined model

Mirror `experiments/eval_ray_attention_temporal_crossview_residual_mpiinf3dhp.py`
for the new model.  This establishes a baseline checkpoint before any GVHMR-style
fine-tuning and lets us compare against the 11.17 mm cross-view residual result.

### 2.3 GVHMR-style fine-tuning

The new model should be fine-tuned on synthetic monocular-to-multi-view data
that mimics GVHMR errors: project clean 3D SMPL joints through a virtual rig and
inject pixel noise, view dropout, and 2D outliers.  Use a low learning rate
(1e-4) and keep the MPI validation guard to avoid catastrophic forgetting.

### 2.4 Advanced GVHMR demo script

Create `experiments/demo_gvhmr_multiview_projection_advanced.py`.  It loads
`hmr4d_results.pt`, runs SMPL forward, projects through virtual cameras (or a
real calibration file), feeds temporal clips to the combined plugin, and reports
both accuracy vs the single-view GVHMR reference and geometric reprojection error.

### 2.5 Temporal smoothing and skeleton constraints

Add a lightweight post-processing stage: 1D Savitzky-Golay smoothing and a soft
bone-length consistency loss.  This directly improves the visual quality of
the demo and gives a paper-ready temporal-consistency metric.

### 2.6 Per-joint uncertainty visualization

The combined model already predicts `log_var`.  Export per-joint uncertainty
heatmaps and per-view weight maps for the demo video frames; this is a strong
qualitative result for the paper.

## 3. Experiments to Run

1. **Benchmark training of the combined model on MPI-INF-3DHP**
   - Use `experiments/train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.py`.
   - Sweep: `d ∈ {64, 128}`, `n_st_layers ∈ {2, 4}`, `residual_hidden ∈ {128, 256}`.
   - Track MPJPE, PA-MPJPE, PCK@50/100/150, AUC on S2/Seq1 validation.

2. **GVHMR-style fine-tuning**
   - Start from the best checkpoint from (1).
   - Generate synthetic projections with `noise_std=2.0` px, `dropout_rate=0.1`,
     `outlier_rate=0.05`.
   - Train 10-20 epochs at LR=1e-4 with reprojection loss weight 0.01.
   - Validate that clean MPI MPJPE stays within 11.0-12.0 mm.

3. **Demo evaluation matrix**
   - Clean (noise 0.5 px), moderate noise (2 px), and challenging (2 px + 10 % dropout + 5 % outliers).
   - Report MPJPE/PA-MPJPE vs the single-view GVHMR world reference and per-view
     reprojection error.

4. **Ablation on the demo set**
   - Disable uncertainty (set `log_var = 0`), Gauss-Newton, and residual head
     separately to isolate gains.

## 4. Metrics to Track

- **Accuracy:** MPJPE, PA-MPJPE (mm) vs GVHMR reference and vs 3D GT when available.
- **Robustness:** PCK@50/100/150, AUC on MPI validation.
- **Geometry sanity:** per-view reprojection error (px).
- **Temporal quality:** mean acceleration error of 3D joints (third finite difference).
- **Per-joint/per-view breakdown:** locate wrists/ankle failures and bad cameras.
- **Uncertainty calibration:** rank correlation between predicted `log_var` and
  actual reprojection error.

## 5. Risks

- **Single-view proxy only:** real synchronized multi-view capture is still
  missing; the demo is a controlled simulation.
- **Catastrophic forgetting:** fine-tuning on synthetic GVHMR-style data may
  degrade MPI benchmark numbers.
- **Memory:** the combined model has the largest attention footprint; long clips
  (≥13 frames) at `d=128` may need gradient checkpointing.
- **SMPL joint mismatch:** GVHMR emits SMPL joints; the model default is 17
  joints.  A 17-joint subset must be selected or the skeleton re-mapped.
- **Missing WebBridge data:** `data/webbridge/mpi_inf_3dhp/*.npz` must be generated
  before the training script can run.

## 6. Code Sketch

```python
# motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_module.py
import numpy as np
import torch
from ..calibration.camera import Camera
from .fusion_module import FusionModule
from .ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model import (
    RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1,
)

class RayAttentionTemporalCrossviewUncertaintyResidualLearnedTriFusionModule(FusionModule):
    name = "ray_attention_temporal_crossview_uncertainty_residual_learned_tri"

    def __init__(self, j=17, d=64, n_views=4, checkpoint_path=None, input_scale=1.0):
        super().__init__()
        self.input_scale = input_scale
        self.model = RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1(
            j=j, d=d, n_views=n_views
        )
        if checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        self.model.eval()

    def fuse(self, points_2d: np.ndarray, confidences: np.ndarray, cameras: list[Camera]):
        # points_2d: (T, V, J, 2), confidences: (T, V, J)
        x = np.concatenate([points_2d, confidences[..., None]], axis=-1)  # (T, V, J, 3)
        x_t = torch.from_numpy(x).float().unsqueeze(0)  # (1, T, V, J, 3)
        K = torch.stack([torch.from_numpy(cam.K) for cam in cameras]).float().unsqueeze(0)
        R = torch.stack([torch.from_numpy(cam.R) for cam in cameras]).float().unsqueeze(0)
        t = torch.stack([torch.from_numpy(cam.t) for cam in cameras]).float().unsqueeze(0)
        with torch.no_grad():
            pred, _, _, _ = self.model(x_t, K=K, R=R, t=t)
        return pred[0].cpu().numpy()  # (T, J, 3)
```

```python
# experiments/demo_gvhmr_multiview_projection_advanced.py (key fragment)
from motionflow_mv.fusion.ray_attention_temporal_crossview_uncertainty_residual_learned_tri_module import (
    RayAttentionTemporalCrossviewUncertaintyResidualLearnedTriFusionModule,
)

plugin = RayAttentionTemporalCrossviewUncertaintyResidualLearnedTriFusionModule(
    j=17, d=64, n_views=4,
    checkpoint_path="outputs/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.pth"
)

pred_3d = plugin.fuse(points_2d, confidences, cameras)  # (T, J, 3)
```

## 7. Next Action Checklist

- [ ] Implement the plugin wrapper and register it in `motionflow_mv/fusion/__init__.py`.
- [ ] Add `experiments/eval_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.py`.
- [ ] Run benchmark training and select the best checkpoint.
- [ ] Create `experiments/train_gvhmr_style_combined_model.py` for fine-tuning.
- [ ] Create `experiments/demo_gvhmr_multiview_projection_advanced.py`.
- [ ] Add temporal smoothing + bone-length post-processing.
- [ ] Generate uncertainty/weight visualizations.

