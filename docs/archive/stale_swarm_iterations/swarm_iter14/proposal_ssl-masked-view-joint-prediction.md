# Proposal: Self-Supervised Multiview Pretext Task — Masked View/Joint Prediction

**Author:** iter14 swarm agent (SSL / data-efficiency track)  
**Date:** 2026-08-06  
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP clean **9.32 mm** MPJPE / **5.37 mm** PA-MPJPE  

---

## 1. Problem (1 sentence)

The existing SSL pretext task in `experiments/pretrain_ray_attention_ssl.py` masks only whole views or time steps, so it cannot learn fine-grained joint-level completion cues that are critical when individual body joints are occluded in downstream inference.

---

## 2. Hypothesis (1 sentence)

If we extend the self-supervised pretext to randomly mask **both views and joints** and train the model to reconstruct the masked 2D observations from the remaining multiview context, the learned representations will transfer better to supervised pose estimation and improve robustness under occlusion.

---

## 3. Method

### 3.1 Data changes

**Modify `motionflow_mv/data/ssl_dataset.py`:**

- Add a `mask_joints()` helper that applies independent per-joint masks to the `(B, T, V, J)` tensor, with at least `ensure_min_joints` visible per frame.
- Extend `MaskedViewReprojectionDataset` with a `mask_joint_ratio` argument and a combined `(view_or_joint_mask)` flag so each batch returns:
  ```
  x_masked, view_mask, joint_mask, x_original, K, R, t
  ```
- Keep the existing `mask_views()` semantics unchanged to avoid breaking the current smoke test.

**No changes to `MixedDataset` are required for the smoke**; the pretext task uses canonical `.npz` files with `points_2d`, `confidences`, `camera_K/R/t` only.

### 3.2 Loss / architecture changes

**Modify `experiments/pretrain_ray_attention_ssl.py`:**

- Add CLI flags:
    - `--mask_view_ratio` (default `0.15`)
    - `--mask_joint_ratio` (default `0.15`)
    - `--mask_mode` (choices: `"view"`, `"joint"`, `"view_joint"`, `"mixed"`)
    - `--lambda_mask_view` / `--lambda_mask_joint` loss weights (defaults `1.0`)
- In the training loop, sample **both** a view mask and a joint mask, zero-out the corresponding confidence values, and forward the masked input through the existing `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`.
- Compute reprojection loss separately on:
    - visible slots,
    - masked-view slots (all joints in those views),
    - masked-joint slots (those joints across all views),
- Keep the existing temporal-smoothness and bone-length regularizers.

**New helper module `motionflow_mv/losses/masked_ssl_loss.py` (optional, minimal):**

A thin wrapper around the existing `reprojection_loss` that accepts `view_mask` and `joint_mask`, splits the loss into visible / view-masked / joint-masked terms, and returns a dict for logging. If this file is overkill, the logic can live inline in the trainer; the proposal keeps it in the trainer for the smoke test.

### 3.3 Trainers / scripts to create

- `experiments/smoke_ssl_masked_view_joint.py` — CPU/GPU smoke script that runs 3 epochs on synthetic 4-view/17-joint data and verifies finite loss and no NaNs.
- `experiments/pretrain_ssl_masked_view_joint.py` — full SSL trainer, initially a thin fork of `pretrain_ray_attention_ssl.py` that exercises the new masking flags.

### 3.4 Exact files touched

| File | Action | What |
|---|---|---|
| `motionflow_mv/data/ssl_dataset.py` | Modify | Add `mask_joints()` and extend `MaskedViewReprojectionDataset` / `collate_masked_fn` to carry `joint_mask`. |
| `experiments/pretrain_ray_attention_ssl.py` | Modify | Add `--mask_view_ratio`, `--mask_joint_ratio`, `--mask_mode`, split masked loss into view/joint terms. |
| `experiments/smoke_ssl_masked_view_joint.py` | Create | 3-epoch smoke on synthetic `.npz`. |
| `experiments/pretrain_ssl_masked_view_joint.py` | Create | Full H36M/MPI SSL trainer for the new pretext. |

---

## 4. Smoke-Test Plan

Use the existing synthetic-data pattern from `experiments/smoke_pretrain_ray_attention_ssl.py`.

**Configuration:**

- Synthetic: 4 views, 17 joints, `clip_len=9`, 80 train frames / 30 val frames.
- Model: `d=32`, `residual_hidden=64`, `n_st_layers=1`.
- Masking: `mask_view_ratio=0.25`, `mask_joint_ratio=0.25`, `mask_mode="view_joint"`.
- Batch size 2, 3 epochs, on CPU or the local RTX 4090.

**Pass / fail criteria:**

- **Pass:** script completes without errors or NaNs in < 10 minutes on the RTX 4090.
- **Pass:** total validation loss decreases monotonically across the 3 epochs.
- **Pass:** masked-joint reprojection loss is finite and non-increasing in the last epoch (model learns to fill masked joints).
- **Fail:** any NaN/Inf, runtime > 15 min, or loss increases in the final epoch.

---

## 5. Evaluation Plan

### 5.1 Pretext metrics

- `val_vis_loss`, `val_view_mask_loss`, `val_joint_mask_loss` logged per epoch.
- Ratio `joint_mask_loss / visible_loss < 2.0` after 10 epochs on H36M (heuristic that masked joints are being reconstructed, not ignored).

### 5.2 Downstream transfer metrics

- Fine-tune the SSL-pretrained backbone on a small supervised split of H36M (e.g., 50 clips).
- Compare against an identical model trained from scratch:
    - MPJPE / PA-MPJPE on the H36M validation set.
    - Target: ≥ 5% relative MPJPE improvement, or any improvement if the baseline is already strong.

### 5.3 Robustness metrics

- Run the 6-axis robustness matrix (`experiments/eval_robustness_matrix_pp_mpiinf3dhp.py`) after fine-tuning.
- Focus on `joint_dropout_0.2` and `joint_dropout_0.4`:
    - Target: joint-dropout degradation is ≤ 20% worse than the clean baseline (i.e., the SSL pretext specifically helps occluded joints).

### 5.4 Scripts

- `experiments/smoke_ssl_masked_view_joint.py`
- `experiments/pretrain_ssl_masked_view_joint.py`
- `experiments/eval_ssl_finetune_masked_joint.py` (thin downstream fine-tune + eval script, reuses `motionflow_mv/eval/benchmark_protocol.py`)

---

## 6. Estimated GPU/CPU Cost on RTX 4090

| Stage | Runtime | Resource |
|---|---|---|
| Smoke (3 epochs, synthetic) | ~1–2 min CPU, <1 min GPU | CPU or RTX 4090 |
| SSL pretrain on H36M (~1.6 M frames, 50 epochs) | ~4–6 h | RTX 4090 |
| Downstream fine-tune (50 clips, 20 epochs) | ~30–60 min | RTX 4090 |
| Robustness matrix smoke (20 clips) | ~10 min | CPU / GPU eval only |

**Memory:** synthetic smoke uses < 2 GB; H36M SSL pretrain uses ~4–6 GB with `d=64`, `batch_size=8`.

---

## 7. Risks & Fallback

| Risk | Impact | Fallback |
|---|---|---|
| Joint masking collapses to mean pose because too many joints are hidden. | High | Lower `mask_joint_ratio` to `0.10` or add a direct per-joint 2D coordinate regression head. |
| Reprojection loss on masked slots is noisy and dominates training. | Medium | Down-weight `lambda_mask_joint` to `0.5` and up-weight visible loss; or add a small SSL projection head. |
| Smoke script is CPU-bound or too slow on RTX 4090 (as happened with visibility v2). | Medium | Reduce `clip_len` to 5 and `train_samples` to 20; keep model at `d=32`. |
| No downstream accuracy gain after SSL pretrain. | High | Treat this as a negative result and pivot to supervised-only joint dropout augmentation (no SSL). |
| Existing `reprojection_loss` does not support per-joint masks cleanly. | Low | Inline the mask-splitting logic in the trainer before calling `reprojection_loss`. |

---

## 8. Integration with Iter13 Anchor

The output of this experiment is a **pretrained checkpoint** that can be loaded into the existing `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` backbone for the supervised trainers (`experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`). If the smoke and transfer eval pass, the pretrained checkpoint becomes an optional `--ssl_checkpoint` argument for the supervised training scripts and is evaluated in the next robustness / SOTA comparison sweep.
