# Iter11+ MPI-INF-3DHP Full-Training Roadmap

## Current State

The project has a strong cross-view residual baseline and a new “all-in-one” fusion model. On the partial MPI-INF-3DHP split (S1/S3 train, S2 val) the current best validation MPJPE is **11.17 mm** (`outputs/eval_residual_final5.json`, `ray_attention_temporal_residual_final5.pth`, 28 joints, 14 views, clip length 13). A lighter cross-view residual run reaches **15.29 mm**, and the temporal-residual + reprojection fast run only reached **~47.5 mm** because it was trained on a small subset for very few epochs. The new combined model, `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1`, has not yet been trained with the full MPI-INF-3DHP training corpus.

Data currently converted: only S1–S3 (`data/webbridge/mpi_inf_3dhp/raw/{S1,S2,S3}`). All existing training scripts accept only one or two sequences and use `RandomClipDataset` with a few thousand clips. The new model adds three extra components on top of the cross-view architecture:

1. **Uncertainty-weighted DLT** (per-view log-variance prediction + NLL loss).
2. **Differentiable Gauss–Newton triangulation** head.
3. **Residual refinement MLP** after the GN-refined 3D estimate.

This report outlines concrete, implementable steps to train that model to ICRA/CVPR 2027 standards on the full MPI-INF-3DHP dataset.

---

## 1. Complete and Standardize the Dataset

### Action
Finish converting the official MPI-INF-3DHP training set. The standard split is:

- **Train:** S1–S8 (sequences 1 and 2, where available).
- **Validation:** S2 Seq1 (the existing held-out sequence) or, if reporting an official test score, keep the test set (`TS1`/`TS2`) separate.

Use the existing batch converter:

```bash
conda run -n mf python experiments/batch_convert_mpiinf3dhp_v1.py \
    --subjects 4 5 6 7 8 \
    --sequences 1 2 \
    --download --yes-download
```

Currently only S1–S3 are on disk; S4–S8 must be downloaded and converted. The converter already emits both 14-view and 4-view variants (`_v14_` and `_v4_`) and meter-scaled versions (`_m.npz`), so no extra preprocessing is needed.

### Proposed Full-Training Wrapper
Create `experiments/train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp_full.py` that feeds the full set of training `.npz` files and a held-out validation file to the combined model:

```python
DATA_DIR = Path("data/webbridge/mpi_inf_3dhp")
TRAIN_FILES = [
    DATA_DIR / f"s_{s:02d}_seq_{q:02d}_v14_multiview_m.npz"
    for s in [1, 3, 4, 5, 6, 7, 8]
    for q in [1, 2]
]
VAL_FILE = DATA_DIR / "s_02_seq_01_v14_multiview_m.npz"

# Example run (full 14-view, scaled-up model):
cmd = [
    sys.executable,
    "experiments/train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.py",
    "--train", *[str(p) for p in TRAIN_FILES],
    "--val", str(VAL_FILE),
    "--clip_len", "13",
    "--d", "128",
    "--n_st_layers", "3",
    "--residual_hidden", "256",
    "--gn_iters", "3",
    "--gn_damping", "1e-6",
    "--uncertainty_weight", "0.1",
    "--reproj_weight", "0.01",
    "--epochs", "100",
    "--lr", "1e-3",
    "--batch_size", "8",
    "--train_samples", "4000",
    "--output", "outputs/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp_full.pth",
]
```

---

## 2. Scale Model Capacity and Training Schedule

The 11.17 mm checkpoint uses `d=64`, `n_st_layers=2`, `residual_hidden=128`. The best cross-view residual run already showed that scaling helps (`d=64, h=128` is not the largest tested). For the combined model:

| Hyper-parameter | Current | Full-training target |
|-----------------|---------|----------------------|
| `d`             | 64      | 128                  |
| `n_st_layers`   | 2       | 3                    |
| `residual_hidden` | 128   | 256                  |
| `n_heads`       | 4       | 8 (with d=128)       |
| Epochs          | 30      | 100                  |
| LR schedule     | constant 1e-3 | cosine annealing, warm-up 5 epochs, decay to 1e-5 |
| Batch size      | 8       | 8–16 (gradient accumulation if memory-limited) |
| Optimizer       | Adam    | AdamW (weight decay 1e-4) |

Use `torch.optim.lr_scheduler.CosineAnnealingLR` or `OneCycleLR`, and add a 5-epoch linear warm-up. This is the same regime that produced the 11.17 mm result, extended to the larger model and longer schedule.

---

## 3. Stronger Augmentation and View Dropout

The current `augment_clip` adds 2D Gaussian noise, confidence dropout, and outlier pixels. For full training, add view-aware augmentation:

```python
def augment_clip_mpi(x, cameras=None, noise_std=1.0, dropout_rate=0.15,
                     outlier_rate=0.03, view_dropout_rate=0.2):
    # x: (B, T, V, J, 3) = (u, v, confidence)
    if noise_std > 0:
        x[..., :2] += torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = torch.rand(x.shape[:-1], device=x.device) > dropout_rate
        x[..., 2] *= mask.float()
    if outlier_rate > 0:
        outlier_mask = torch.rand(x.shape[:-1], device=x.device) < outlier_rate
        outlier = (torch.rand(x.shape[:-1] + (2,), device=x.device) - 0.5) * 200.0
        x[..., :2] = torch.where(outlier_mask[..., None], outlier, x[..., :2])
    # Randomly drop entire views to improve robustness to occlusion.
    if view_dropout_rate > 0 and x.shape[2] > 4:
        for b in range(x.shape[0]):
            if random.random() < view_dropout_rate:
                v = random.randint(0, x.shape[2] - 1)
                x[b, :, v, :, 2] = 0.0
    return x
```

Additional augmentations to test:

- **Temporal jitter:** randomly sub-sample / repeat frames within the clip to simulate variable frame rates.
- **2D rotation / scaling:** rotate or scale all 2D keypoints in a view around the image center.
- **Bone-length regularization:** add an auxiliary loss encouraging predicted skeletons to have plausible bone lengths.

---

## 4. Auxiliary Losses

The combined model already returns `nll_loss` and the training script can add a reprojection loss (`--reproj_weight`). For full training, combine:

1. **3D MPJPE / MSE** against ground truth (main loss).
2. **Reprojection loss** (`motionflow_mv.losses.reprojection_loss`) weighted by `reproj_weight=0.01`.
3. **Uncertainty NLL** (`nll_loss` returned by the model, weight `0.1`).
4. **Temporal / velocity smoothness loss** on predicted 3D sequences.
5. **Bone-length loss** (optional) to penalize unrealistic limb lengths.

The total loss should be:

```python
loss = mse(pred_3d, gt_3d) \
     + λ_reproj * reprojection_loss(pred_3d, points_2d, K, R, t, conf) \
     + nll_loss \
     + λ_vel * velocity_loss(pred_3d)
```

Start with `λ_reproj=0.01`, `λ_vel=0.001`, and tune on the validation set.

---

## 5. Transfer from Human3.6M WebBridge

H36M WebBridge conversion is in progress at `data/webbridge/h36m`. Once available, pre-train the combined model on H36M multi-view data (17 joints) and then fine-tune on MPI-INF-3DHP. Even with different skeletons, the model’s early view/camera encoding layers transfer well. Implementation:

```bash
# 1. Pre-train on H36M WebBridge
python experiments/train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_h36m.py \
    --train data/webbridge/h36m/s_01_acts_*_multiview_m.npz ...

# 2. Fine-tune on MPI, allowing only residual/uncertainty heads to update for 5 epochs.
python experiments/train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp_full.py \
    --checkpoint outputs/pretrain_h36m.pth \
    --freeze_encoder 5
```

If skeletons differ, map the common joints or simply treat the H36M pretraining as warm-up for the encoder and re-initialize the final joint-specific heads for MPI.

---

## 6. Test-Time Refinement and Ensembling

After training, two low-effort boosts are available:

- **Sliding-window ensemble:** evaluate with overlapping clips and average predictions in the overlap region.
- **Test-time Gauss–Newton:** run extra GN iterations at inference (the model already supports `gn_iters`); increasing from 3 to 5–10 may refine hard joints.
- **Model ensemble:** train 3–5 models with different seeds and average their predictions; this typically reduces MPJPE by 5–10%.

---

## 7. Experiments to Run and Metrics to Track

Run the following experiments in this order:

| # | Experiment | What to measure |
|---|-----------|-----------------|
| 1 | Baseline combined model on S1+S3, 30 epochs, `d=64` | MPJPE, PA-MPJPE, PCK@50/100/150, AUC |
| 2 | Combined model on **full S1–S8**, `d=128`, 100 epochs | Same + per-joint errors |
| 3 | Effect of view dropout + stronger augmentation | Delta MPJPE on val |
| 4 | Add reprojection + velocity losses | Delta MPJPE, convergence speed |
| 5 | H36M pre-train → MPI fine-tune | MPJPE vs. from-scratch |
| 6 | Test-time ensemble + sliding-window averaging | Final test-set MPJPE |

Primary metrics: **MPJPE (mm)**, **PA-MPJPE (mm)**, **PCK@50/100/150 mm**, and **AUC**. Track per-joint errors to identify whether wrists/ankles remain the bottleneck.

Use Weights & Biases or TensorBoard for live loss curves, learning rate, and validation MPJPE. Set the config `configs/train_ray_attention_reproducible.yaml` fields (`wandb.enabled: true`) to enable logging.

---

## 8. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| S4–S8 raw data not yet downloaded / converted | High | Run `batch_convert_mpiinf3dhp_v1.py --download --yes-download` for subjects 4–8 first. |
| GPU memory with `d=128`, 14 views, clip=13 | Medium | Use gradient accumulation, mixed precision (`torch.cuda.amp`), or reduce batch size to 4. |
| Differentiable Gauss–Newton becomes unstable | Medium | Keep damping at `1e-6`; clip gradients; optionally clamp `log_var`. |
| Overfitting on small MPI train set | Medium | Use view dropout, temporal augmentation, and H36M pretraining. |
| Long training time | Medium | Train on the smaller `d=64` model first as a smoke test, then scale up. |
| Test-set leakage | Low | Hold out S2 (and TS1/TS2) explicitly; never train on them. |

---

## Summary of Next Steps

1. Download and convert **S4–S8** to unlock the full MPI-INF-3DHP training corpus.
2. Create the full-training wrapper for the combined model with `d=128`, `n_st_layers=3`, `residual_hidden=256`, 100 epochs, and a cosine LR schedule.
3. Add view dropout, temporal jitter, and stronger 2D augmentation.
4. Add reprojection and velocity auxiliary losses to the combined loss.
5. Pre-train on H36M WebBridge, then fine-tune on MPI.
6. Evaluate with MPJPE/PA-MPJPE/PCK/AUC, track per-joint errors, and apply test-time ensembling.

This sequence should push the current **~11.17 mm** validation MPJPE substantially lower and produce a publishable MPI-INF-3DHP result.
