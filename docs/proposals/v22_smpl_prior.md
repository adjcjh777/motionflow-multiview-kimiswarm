# V22 SMPL-Aware Fusion Prior

**Task identifier:** `design_v22_smpl_prior`

## Motivation

Recent work in the repository has shown that camera-conditioned ray attention,
robust triangulation, and residual refinement are effective for multi-view 3D
pose estimation.  However, the current models estimate joints independently and
do not explicitly enforce anthropometric consistency.  Adding a parametric human
body prior (SMPL) can:

* regularize implausible bone lengths and joint configurations,
* provide a complete body representation that can be rendered/re-projected,
* give a compact, interpretable output in the form of SMPL shape/pose
  parameters, which is useful for downstream animation and motion-capture
  applications.

This proposal introduces an `SMPLPriorFusionV22` module, which augments
`OmniMultiViewFusionV5` with a small SMPL head and uses the resulting 3D body
as a prior for the final 3D joint estimates.

## Design

### Module: `motionflow_mv/fusion/smpl_prior_fusion_v22.py`

Two classes are provided:

1. `SMPLPriorHead`
   * Consumes the concatenation of per-joint pooled features and the raw
     triangulated 3D pose, i.e. `(N, J, d + 3)`.
   * Predicts shared clip-level shape `betas` (`1 x 10`).
   * Predicts per-frame `body_pose` (`N x 69`), `global_orient` (`N x 3`), and
     `transl` (`N x 3`).
   * Predicts a per-frame scalar blend weight in `[0, 1]` via sigmoid.
   * When `smplx` is installed and a model path is supplied, runs the SMPL
     forward pass and returns `smpl_joints` and `pred_joints_17`.

2. `SMPLPriorFusionV22`
   * Subclasses `OmniMultiViewFusionV5`.
   * Registers a forward hook on the base model's `residual_mlp` to capture its
     input without duplicating the base forward pass.
   * Feeds that input to `SMPLPriorHead`.
   * When SMPL joints are available, blends them with the triangulation-based
     estimate using the learned blend weight.
   * Returns the standard v5 tuple by default, and appends the SMPL output
     dictionary when `return_smpl=True`.

### Blend operation

Given the triangulation-based prediction `pred_3d` and the SMPL prior joints
`J_smpl`, the final prediction is:

```
pred_3d = (1 - alpha) * pred_3d + alpha * J_smpl
```

where `alpha = sigmoid(blend)` is learned per frame.  This keeps the model
faithful to observations when it is confident, while allowing the anthropometric
prior to dominate for noisy/occluded views.

### Optional base-model freezing

`freeze_base=True` sets `requires_grad=False` for all base v5 parameters and
keeps only the SMPL head trainable.  This is useful for diagnosing whether the
SMPL head can be trained in isolation and for staged-fine-tuning ablations.

## Integration

```python
from motionflow_mv.fusion.smpl_prior_fusion_v22 import SMPLPriorFusionV22

model = SMPLPriorFusionV22(
    j=17,
    d=64,
    n_views=4,
    smpl_model_path="data/smpl/SMPL_NEUTRAL.pkl",
)

# Standard v5 outputs + SMPL dictionary.
pred_3d, weights, visibility, L, epi_loss, smpl_out = model(
    x, cameras=cameras, return_smpl=True
)
```

When `smplx` is unavailable, the module still predicts parameters but does not
run the parametric body; the blend weight becomes a residual-only placeholder
that is learned end-to-end.

## Testing

A pytest file is added at `tests/test_smpl_prior_fusion_v22.py` that covers:

* Correct output shapes for the SMPL prior head.
* Forward and backward pass of the full `SMPLPriorFusionV22` model.
* The `return_smpl=True` path.
* The `freeze_base=True` behavior.

Run with:

```bash
pytest tests/test_smpl_prior_fusion_v22.py -v
```

## Future Work / Open Questions

* **Joint regressor:** The current implementation maps the first 17 SMPL joints
  directly to the H36M-style skeleton.  A learned joint regressor or a
  bone-length loss could improve accuracy.
* **Bone-length regularizer:** Add a soft constraint that the SMPL body and the
  predicted 3D joints share the same bone-length distribution.
* **Temporal consistency:** SMPL parameters can be temporally smoothed (e.g.
  `transl` and `body_pose` temporal consistency losses).
* **Training strategy:** In the first stage, the SMPL head can be trained with
  ground-truth 3D pose supervision; in the second stage, the blend weight can be
  learned jointly with the base model.

## References

* Loper et al., "SMPL: A Skinned Multi-Person Linear Model", SIGGRAPH Asia 2015.
* Existing multi-task SMPL prototype: `motionflow_mv/fusion/multi_task_shape_pose.py`.
