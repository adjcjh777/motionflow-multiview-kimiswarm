# Design: Domain-Adaptive Wrapper for Synthetic-to-Real Transfer (task_14)

**Author:** Swarm agent task_14
**Date:** 2026-08-05
**Related files:**
- `motionflow_mv/fusion/domain_adaptation_wrapper.py`
- `experiments/train_domain_adapt_mpiinf3dhp.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_model.py` (read-only, base model)
- `motionflow_mv/fusion/ray_attention_temporal_model_domain.py` (prior camera-domain work)
- `motionflow_mv/fusion/ray_attention_v3_model.py` (prior GRL / domain-agnostic camera code)

## Goal

Build a minimal, working **domain-adaptive wrapper** around the current best
`RayAttentionFusionModelTemporalResidual` backbone so that a model pre-trained
on synthetic multi-view pose data can be fine-tuned toward real-world data
(MPI-INF-3DHP) without modifying the backbone itself.  The deliverable is a
new wrapper module plus a training script, validated by CPU/small-data smoke
tests only.

## Motivation

The project has strong single-dataset numbers on MPI-INF-3DHP (10.46 mm MPJPE
with the residual model), but synthetic-to-real transfer remains a gap:

- Synthetic data (SMPL/AMASS renderings) is cheap and large but has a different
  noise distribution, camera baseline, and joint appearance.
- Real data (e.g., MPI-INF-3DHP) is expensive to annotate and often scarce.

A domain-adaptive wrapper should let us exploit synthetic labels while
encouraging the shared encoder to produce domain-invariant features, and while
giving the model just enough domain-specific capacity to handle residual
appearance shifts.

## Design

### Architecture

`DomainAdaptationWrapper` (in `motionflow_mv/fusion/domain_adaptation_wrapper.py`)
wraps `RayAttentionFusionModelTemporalResidual` and adds two optional,
learnable mechanisms:

1. **Gradient-Reversal Domain Discriminator**
   - A small MLP classifier is attached to the pooled temporal features.
   - A `GradientReversalLayer` multiplies gradients by `-λ` before the
     classifier, forcing the shared encoder to produce features that confuse
     the discriminator (domain-invariant features).
   - Output is `(B, 2)` logits (synthetic vs. real).

2. **Domain-Specific FiLM Adapters**
   - Each domain (0 = synthetic, 1 = real) has a small linear layer that
     predicts affine parameters γ and β from a learnable domain embedding.
   - The affine parameters modulate the temporal features before the weight
     head and residual head.
   - This lets the model retain domain-specific statistics without changing
     the shared encoder weights.

The wrapper subclasses the residual model and copies only the forward logic,
so the base model file is left untouched and other swarm members can continue
working on it independently.

### Training Objective

The training script (`experiments/train_domain_adapt_mpiinf3dhp.py`) mixes
synthetic and real clips in each batch.  Loss terms:

- **Pose regression loss:** MSE between predicted and ground-truth 3D joints,
  computed only on labeled clips (synthetic by default; real too if labels are
  available).
- **Domain adversarial loss:** Cross-entropy on the GRL domain logits,
  weighted by `λ_domain`.
- **MMD (optional):** A helper `maximum_mean_discrepancy()` is provided for
  feature-distribution alignment, but it is disabled by default
  (`λ_mmd=0`).

The script also supports `--unlabeled_real`, in which case real clips are only
used for the domain-adversarial objective.

### Data Format

Same canonical `.npz` layout used by the rest of the temporal scripts:

```
points_2d   (T, V, J, 2)
confidences (T, V, J)
joints_3d   (T, J, 3)
camera_K    (V, 3, 3)
camera_R    (V, 3, 3)
camera_t    (V, 3)
```

Synthetic and real files are passed separately on the command line; the dataset
class samples clips from both with roughly balanced domain labels.

## Expected Impact

- **Primary:** Provides a reusable, non-invasive way to fine-tune the residual
  model on real data when only synthetic labels are abundant.
- **Secondary:** The GRL branch should reduce domain-specific overfitting during
  synthetic pre-training; the FiLM adapters give a modest capacity boost for
  real-world appearance/noise differences.
- **Risk:** Domain-adversarial training can be unstable; the wrapper keeps the
  discriminator small and λ_domain configurable so the effect can be dialed in
  during full training.

## Files Created / Modified

- **New:** `motionflow_mv/fusion/domain_adaptation_wrapper.py`
- **New:** `experiments/train_domain_adapt_mpiinf3dhp.py`
- **New:** `docs/swarm_iter_next/design_domain_adaptation/design_domain_adaptation.md`
- **New (smoke helper):** `tmp/gen_smoke_data.py`
- **New (smoke outputs):** `tmp/smoke_data/{synthetic,real,val}.npz`
- **Modified:** none of the existing project files.

## How to Test / Validate

1. **Module sanity check**
   ```bash
   KMP_DUPLICATE_LIB_OK=TRUE /d/anaconda3/python.exe -m motionflow_mv.fusion.domain_adaptation_wrapper
   ```
   Expected output: `domain adaptation wrapper sanity check passed`

2. **Smoke training (labeled real)**
   ```bash
   # generate tiny data once
   KMP_DUPLICATE_LIB_OK=TRUE /d/anaconda3/python.exe tmp/gen_smoke_data.py

   KMP_DUPLICATE_LIB_OK=TRUE /d/anaconda3/python.exe experiments/train_domain_adapt_mpiinf3dhp.py \
       --synthetic_train tmp/smoke_data/synthetic.npz \
       --real_train tmp/smoke_data/real.npz \
       --val tmp/smoke_data/val.npz \
       --clip_len 9 --epochs 2 --batch_size 2 --train_samples 20
   ```
   Expected: script runs for 2 epochs and reports a best validation MPJPE.

3. **Smoke training (unlabeled real)**
   ```bash
   KMP_DUPLICATE_LIB_OK=TRUE /d/anaconda3/python.exe experiments/train_domain_adapt_mpiinf3dhp.py \
       --synthetic_train tmp/smoke_data/synthetic.npz \
       --real_train tmp/smoke_data/real.npz \
       --val tmp/smoke_data/val.npz \
       --clip_len 9 --epochs 1 --batch_size 2 --train_samples 20 \
       --unlabeled_real
   ```
   Expected: script runs and uses domain loss on real clips while computing
   pose loss only on synthetic clips.

## Blockers / Next Steps

- **Data availability:** The smoke data is purely random; for real validation,
  the project needs a canonical synthetic `.npz` (SMPL/AMASS rendered) that
  matches the MPI-INF-3DHP skeleton and camera layout.  This can be produced by
  `experiments/generate_synthetic_multiview_dataset.py` or a follow-up task.
- **Full training:** Only CPU smoke tests were run.  GPU-scale experiments are
  needed to measure whether the domain-adversarial term improves real-world
  MPJPE and to tune `λ_domain`, `grl_lambda`, and the FiLM adapters.
- **Evaluation:** A dedicated cross-domain evaluation script (train on synthetic,
  test on MPI-INF-3DHP) would make the impact explicit.
- **Integration:** If proven useful, the wrapper can be registered as a
  `FusionModule` in `motionflow_mv/fusion/__init__.py` so the pipeline can load
  it from a checkpoint.

## References

- Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation", ICML 2015.
- Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018.
- Project prior art: `ray_attention_temporal_model_domain.py` (camera-domain MLP)
  and `ray_attention_v3_model.py` (GRL + domain-agnostic camera features).
