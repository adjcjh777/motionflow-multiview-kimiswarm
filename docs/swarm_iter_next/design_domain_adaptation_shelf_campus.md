# Design: Domain Adaptation for Shelf/Campus

**Author:** domain_adaptation_shelf_campus agent  
**Date:** 2026-08-06  
**Related:** issues #23, #25  

## Goal

Build a Shelf/Campus-specific domain-adaptation pipeline that takes a model
trained on labelled source rigs (e.g. H36M / MPI-INF-3DHP) and fine-tunes it
on the target Shelf and Campus multi-view rigs, whose camera counts differ from
the source rig.

## Motivation

- The project currently reaches ~9.81 mm clean MPJPE on MPI-INF-3DHP with
  Bayesian Triangulation and ~10.23 mm with Hierarchical Attention, against an
  8.75 mm anchor.
- Shelf and Campus provide real-world target sequences (3 and 5 calibrated
  cameras, 17-joint skeleton) but have very different camera baselines,
  backgrounds, and noise distributions compared with H36M/MPI.
- A generic GRL/FiLM wrapper already exists; the missing piece is handling the
  **mismatched view count** and adding a **self-supervised geometry loss** on
  the unlabeled target domain.

## Proposed Method

### 1. Model variant: `ShelfCampusDomainAdaptationWrapper`

New file: `motionflow_mv/models/shelf_campus_domain_adaptation.py`

```python
class ShelfCampusDomainAdaptationWrapper(DomainAdaptationWrapper):
    ...
    def forward(..., active_views: Optional[Union[int, torch.Tensor]] = None):
        # Pad source clip to target view count with zero-confidence
        # dummy views and duplicated valid camera matrices.
        ...

    def target_reprojection_loss(self, pred_3d, x, K, R, t, ...):
        # Project predicted 3D back into each target view using the
        # corrected intrinsics and compare with input 2D keypoints.
        ...
```

Key points:
- Inherits the existing GRL domain discriminator and FiLM adapters from
  `DomainAdaptationWrapper`, so no backbone changes are required.
- Adds variable-view padding helper `_pad_views_with_valid_cameras` that
  duplicates the last real camera for dummy views, keeping `K` invertible
  while zero-confidence masking makes them invisible to triangulation.
- Exposes `target_reprojection_loss` so the training loop can supervise the
  model on target data even when 3D labels are absent.

### 2. Training script

New file: `experiments/train_domain_adapt_shelf_campus.py`

Training loop:
- Load one or more source `.npz` files and one target `.npz` file.
- Use separate source/target mini-batches to avoid mixing different view counts
  in a single tensor.
- Pad the source batch to the target view count with `_pad_views_with_valid_cameras`.
- Loss = supervised MSE on source + (optional) target labels
         + `lambda_domain` * GRL cross-entropy
         + `lambda_reproj` * target reprojection loss.
- Validation on the target domain.

### 3. Smoke test

New files:
- `experiments/smoke_domain_adapt_shelf_campus.py`
- `scripts/run_domain_adapt_shelf_campus_smoke_wsl.sh`

The smoke test generates synthetic 4-view source and 5-view target `.npz`
files and runs two short epochs to verify that the mismatched view counts do
not crash.  CPU runtime is <2 minutes on the RTX 4090 WSL box.

## Expected Impact

- Provides a reusable wrapper for any cross-rig domain-adaptation scenario,
  not only Shelf/Campus.
- The self-supervised reprojection loss lets the model learn target-camera
  geometry without target 3D labels.
- Risk: GRL training can be unstable; the `lambda_domain` and `lambda_reproj`
  hyperparameters need tuning during a full GPU run.

## How to run

CPU smoke test:

```bash
bash scripts/run_domain_adapt_shelf_campus_smoke_wsl.sh
```

Or directly:

```bash
python experiments/smoke_domain_adapt_shelf_campus.py
```

Full run example (GPU):

```bash
python experiments/train_domain_adapt_shelf_campus.py \
  --source_train data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz \
  --target_train data/shelf_campus/Shelf_Seq1/pseudogt_m.npz \
  --val data/shelf_campus/Shelf_Seq1/pseudogt_m.npz \
  --clip_len 13 --d 64 --residual_hidden 128 --n_st_layers 2 \
  --batch_size 8 --train_samples 2000 --epochs 30
```

## Files to add/modify

- `motionflow_mv/models/shelf_campus_domain_adaptation.py` (new)
- `motionflow_mv/models/__init__.py` (register `ShelfCampusDomainAdaptationWrapper`)
- `experiments/train_domain_adapt_shelf_campus.py` (new)
- `experiments/smoke_domain_adapt_shelf_campus.py` (new)
- `scripts/run_domain_adapt_shelf_campus_smoke_wsl.sh` (new)

## Next concrete validation step

1. Merge the code on a feature branch.
2. Run the CPU smoke test; it should complete in <2 minutes.
3. Run a GPU full run on Shelf or Campus with `lambda_reproj=0.1` and measure
   target-domain MPJPE versus the source-only baseline.
4. If the target MPJPE improves by >0.5 mm, integrate the wrapper into the
   main `MultiViewFusionPlugin` registry.
