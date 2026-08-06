# Paper Story & Ablation Design: Bayesian Triangulation Components

## Motivation

The current best converged result on MPI-INF-3DHP is the principal-point
residual model at **8.75 mm** clean MPJPE.  The Tier-2 Bayesian triangulation
proposal recently reached **9.81 mm** by adding an 2-D covariance, an adaptive
Gauss-Newton refinement step, and an epipolar consistency loss.

To close the remaining gap and to tell a clean paper story, we need to
understand which of these three new components actually matter.  This ablation
isolates each component so the paper can report per-component contribution in a
rigorous ablation table.

## Ablation Questions

1. **Adaptive Gauss-Newton (GN) refinement**: does the learned per-joint
   damping Gauss-Newton step improve over the raw weighted-DLT estimate?
2. **Anisotropic covariance**: does predicting a full 2x2 Cholesky factor
   outperform a simpler isotropic covariance?
3. **Epipolar consistency loss**: does the auxiliary epipolar term help
   convergence, or does it just add regularization noise?

## Implementation

Two new ablation flags are added to the Bayesian model:

- `use_adaptive_gn: bool = True` - when False, the adaptive GN step is skipped.
- `anisotropic_covariance: bool = True` - when False, the covariance head
  predicts a single scalar for isotropic 2-D covariances.

The trainer exposes both flags as CLI arguments and passes them to the model.

A CPU smoke test instantiates the model with every combination of the two flags
and runs a single forward/backward step.

A reusable ablation script trains four variants on MPI-INF-3DHP smoke data and
produces a markdown report for the paper's ablation table:

| Variant | What is removed |
|---------|-----------------|
| full | nothing (baseline) |
| no_adaptive_gn | adaptive Gauss-Newton refinement |
| isotropic_cov | anisotropic covariance |
| no_epipolar | auxiliary epipolar consistency loss |

## How to Run

```bash
# CPU smoke test (<2 min)
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_bayesian_tri_ablation.py -v

# Smoke ablation (2 epochs, tiny data)
python experiments/ablate_bayesian_tri_components.py --smoke --variant full

# Full ablation on MPI-INF-3DHP smoke files (GPU)
python experiments/ablate_bayesian_tri_components.py --epochs 10
```

## References

- Issue #23: multi-view fusion robustness improvements
- Issue #25: paper story and ablation design
- Model: `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py`
- Trainer: `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
- Smoke test: `tests/test_bayesian_tri_ablation.py`
- Ablation script: `experiments/ablate_bayesian_tri_components.py`
