# Multi-View Synchronised Temporal Jitter Augmentation

## Motivation

Current training augmentation in the ray-attention pipeline treats 2D
keypoint noise as independent per-joint and per-view.  Real-world capture
systems, however, exhibit errors that are **shared across all joints within a
camera** (small camera translation/rotation or calibration drift) and
**temporally coherent** over short windows (vibration, rolling shutter,
inter-frame smoothing).  A model trained only on high-frequency independent
noise can overfit to unrealistic statistics and may be brittle to these
correlated perturbations.

This proposal introduces a lightweight augmentation that mimics the structure
of real multi-view failures: per-view affine jitter held constant over short
temporal sub-clips, plus a small independent per-joint noise term.  The goal
is to improve the robustness of the fusion head and help push clean MPJPE
below the 8.75 mm anchor on MPI-INF-3DHP / Human3.6M / WebBridge.

## Method

### 1. `MultiViewSyncAugmentation` (`motionflow_mv/data/multiview_temporal_jitter.py`)

Input: a tensor `x` of shape `(B, T, V, J, C)` where the last channel is
`[u, v, confidence]`.

For each temporal sub-clip (default length 5):

* Sample a per-view 2D translation `t_xy ~ N(0, translation_std * I)`.
* Sample a per-view in-plane rotation `θ ~ N(0, rotation_std_deg)`.
* Sample a per-view log-scale factor `s ~ LogNormal(0, scale_std)`.
* Apply the same transform to **all joints** in that view and sub-clip.
* Add a small independent Gaussian noise per joint.

Optionally perform view dropout while guaranteeing at least `min_views`
remain visible.

All randomness is inside `torch.no_grad()` and the module has no learnable
parameters.

### 2. `MultiViewDataAugmentationWrapper` (`motionflow_mv/models/data_augmentation_multiview_wrapper.py`)

A thin `nn.Module` wrapper that applies `MultiViewSyncAugmentation` to the
input during training, then forwards the jittered input to the base fusion
model.  In evaluation mode the augmentation is bypassed.

Key properties:

* State-dict operations are delegated to the wrapped model, so checkpoints are
  interchangeable with the unwrapped model.
* The wrapper is model-agnostic and only requires `x` as the first positional
  argument.

### 3. Training scripts

* `experiments/train_multiview_sync_aug_smoke.py` – CPU-only smoke run on a
  small synthetic sequence; verifies the wrapper trains and evaluates without
  crashing.
* `experiments/train_multiview_sync_aug_full_mpiinf3dhp.py` – full GPU run
  on MPI-INF-3DHP.  It monkey-patches the base temporal PP model class so the
  augmentation is inserted without modifying the shared training script.
* `scripts/run_multiview_sync_aug_smoke_wsl.sh` and
  `scripts/run_multiview_sync_aug_full_wsl.sh` – WSL entry points.

### 4. Smoke tests

`tests/test_data_augmentation_multiview.py` checks:

* Shape preservation under augmentation.
* Training-mode-only activation (eval returns the input unchanged).
* View dropout enforces `min_views`.
* The wrapper delegates `state_dict` correctly.

## Expected outcome

The hypothesis is that by exposing the model to structured, camera-like
perturbations during training, the learned attention weights and residual
refiner become more robust to small calibration drift and view noise.  This
should lower MPJPE on the clean validation set and, more noticeably, improve
robustness on corrupted / noisy views.  The experiment is low risk because the
wrapper can be disabled at inference and checkpoints remain compatible with
the base model.

## How to run

CPU smoke test (< 2 min on a laptop):

```bash
bash scripts/run_multiview_sync_aug_smoke_wsl.sh
```

Full MPI-INF-3DHP run (GPU):

```bash
bash scripts/run_multiview_sync_aug_full_wsl.sh
```

## Next validation step

Run the full training on the A800 or local RTX 4090 and compare the best
validation MPJPE against the Bayesian Triangulation baseline (9.81 mm).  If
improvement is observed, sweep the augmentation hyperparameters
(`translation_std`, `rotation_std`, `subclip_len`) and apply the wrapper to
the Hierarchical Attention model as well.
