# v33 Knowledge Distillation from Monocular Pretrained Models

> **Slug:** `knowledge_distillation`  
> **Title:** Knowledge distillation from monocular pretrained models  
> **Target:** ICRA/CVPR 2027 — improve OmniMultiViewFusionV5 accuracy/robustness by distilling a pretrained monocular 3-D pose estimator into the multi-view fusion pipeline.  

## 1. Problem statement and motivation

The current best multi-view pipeline, `OmniMultiViewFusionV5` (`motionflow_mv/fusion/omniview_fusion_v5.py`), fuses calibrated multi-view 2-D keypoints but only uses 2-D observations and camera geometry. It ignores the rich single-view 3-D prior that modern monocular pose estimators (e.g. CLIFF, PARE, ScoreHMR, or off-the-shelf H36M-pretrained 2-D-to-3-D lifters) provide. When only a few views are available, occlusions are severe, or calibration is noisy, the multi-view triangulation has no anatomical or scene-level prior to fall back on.

**Hypothesis:** Distilling per-view monocular 3-D predictions into the multi-view student can

1. Reduce clean MPJPE, especially on hard joints (wrists/ankles).
2. Improve variable-view robustness (k = 2..V).
3. Add an interpretable per-view uncertainty that the model already knows how to use via its visibility/covariance heads.

The codebase already has a distillation scaffold (`experiments/train_distilled_student_pp_mpiinf3dhp.py`, `motionflow_mv/models/distilled_student_principal_point_model.py`, `tests/test_distilled_student_pp.py`), but it distills from a *multi-view* Bayesian-triangulation teacher. The v33 direction is the complementary case: the teacher is a **monocular pretrained model** run independently on each view, and the student remains the multi-view fusion model.

## 2. Proposed architecture changes

### 2.1 New module: `MonocularDistillationTeacher`

- **Path:** `motionflow_mv/fusion/monocular_distillation_teacher.py`
- **Role:** Wrapper around a pretrained monocular 3-D pose estimator.
- **Inputs:** `(B, T, V, J, 2)` keypoints + optional RGB crops (if available).
- **Outputs:**
  - `mono_pose_3d`: `(B, T, V, J, 3)` per-view monocular 3-D estimate.
  - `mono_uncertainty`: `(B, T, V, J)` or `(B, T, V, J, 3)` per-joint confidence.
- **Implementation options:**
  - **Option A (lightweight, no extra deps):** Use the same `OmniMultiViewFusionV5` as the teacher, but feed it a *single view* at a time (V=1). This is zero-extra-dependency and immediately smoke-testable.
  - **Option B (stronger prior):** Wrap a pretrained CLIFF/PARE/ScoreHMR checkpoint; only if weights already exist in `data/smpl/` or the user provides them.
  - **Recommendation:** Start with **Option A** and treat Option B as a follow-up ablation.

### 2.2 New module: `MonocularDistillationLoss`

- **Path:** `motionflow_mv/losses/monocular_distillation_loss.py`
- **Loss form:**

  ```
  L_distill = α * MSE(pred_3d, mono_mean_3d) + β * KL(student_pose || mono_distribution)
            + γ * MSE(per_view_weights, mono_confidence)
  ```

  - `α`: pose hard-target weight (default 0.1).
  - `β`: distribution-matching weight (default 0.05).
  - `γ`: view-confidence alignment weight (default 0.01).
  - The monocular mean is re-centered via Procrustes to the multi-view world frame before matching.

### 2.3 Model flag additions in `OmniMultiViewFusionV5`

Add to `build_model_from_args` in `experiments/train_omniview_fusion_v5_webbridge_multi.py` and to the `OmniMultiViewFusionV5.__init__` signature:

- `use_monocular_distillation_v33`: bool (default `False`).
- `v33_teacher_type`: str, choices `{"single_view_self", "cliff", "pare", "scorehmr"}` (default `"single_view_self"`).
- `v33_distill_alpha`: float (default 0.1).
- `v33_distill_beta`: float (default 0.05).
- `v33_distill_gamma`: float (default 0.01).
- `v33_procrustes_align`: bool (default `True`).
- `v33_use_uncertainty`: bool (default `True`).
- `v33_freeze_teacher`: bool (default `True`).

The `forward` return tuple is extended only when `use_monocular_distillation_v33=True`:

```python
out += (mono_pose_3d, mono_uncertainty)
```

If the flag is off, the return signature and downstream losses remain **unchanged**, preserving checkpoint compatibility.

### 2.4 Training-script integration

In `experiments/train_omniview_fusion_v5_webbridge_multi.py`:

1. Parse the new flags in `parse_args`.
2. Wire the teacher in `build_model_from_args` via `model_kwargs`.
3. In `compute_loss`:
   - If `args.use_monocular_distillation_v33`:
     - Run a second `with torch.no_grad():` forward on the teacher (or on the same model with V=1).
     - Compute `MonocularDistillationLoss` and add to `loss`.
     - Add `distill_loss` to `metrics` dict.
4. The existing variable-view, outlier-view, and physical-space losses remain active.

### 2.5 Data / preprocessing

No new dataset is required for **Option A**. The WebBridge `.npz` files already contain `points_2d`, `confidences`, `joints_3d`, and calibrated cameras.

For the single-view self-teacher:

- During teacher forward, feed each view independently:
  - `x_v = x[:, :, v:v+1, :, :]` (single active view).
  - Construct a `view_mask` that masks out all other views.
  - Teacher returns a per-view 3-D estimate; stack across views to get `(B, T, V, J, 3)`.

For **Option B** external monocular models:

- Required: RGB crops per view aligned to the 2-D keypoints.
- If not available, fall back to Option A.

## 3. Training command / ablation flags

### Smoke test (CPU/WSL)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --smoke \
  --use_monocular_distillation_v33 \
  --v33_teacher_type single_view_self \
  --v33_distill_alpha 0.1 \
  --v33_distill_beta 0.0 \
  --v33_distill_gamma 0.0 \
  --v33_procrustes_align
```

### Full A800 / RTX 4090 run on WebBridge mixed dataset

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --use_mixed_loader \
  --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
  --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight --use_domain_embedding \
  --use_deformable_cross_view_attention_v18 \
  --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2 \
  --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment \
  --use_hierarchical_multiview_v30 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1 \
  --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 \
  --use_monocular_distillation_v33 \
  --v33_teacher_type single_view_self \
  --v33_distill_alpha 0.1 \
  --v33_distill_beta 0.05 \
  --v33_distill_gamma 0.01 \
  --v33_procrustes_align \
  --v33_use_uncertainty \
  --num_workers 4 --d 64 --residual_hidden 128 --n_st_layers 2 \
  --clip_len 9 --epochs 20 --batch_size 8 --train_samples 1000 --val_stride 10 \
  --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 \
  --output outputs/omniview_fusion_v33_knowledge_distillation.pth
```

### Ablation flags to sweep

| Flag | Default | Sweep |
|------|---------|-------|
| `v33_distill_alpha` | 0.1 | 0.0, 0.05, 0.1, 0.2 |
| `v33_distill_beta` | 0.05 | 0.0, 0.01, 0.05, 0.1 |
| `v33_distill_gamma` | 0.01 | 0.0, 0.005, 0.01, 0.02 |
| `v33_teacher_type` | `single_view_self` | `single_view_self`, `cliff` (if weights available) |
| `v33_procrustes_align` | True | True / False |

## 4. Expected metrics and baseline to beat

### Baseline

Use the v30/v31/v32 best mixed-dataset run. From `scripts/launch_v32_a800_queue.py`, the current strongest single-run config uses:

- `use_multiview_geometry_fusion_v25`
- `use_hierarchical_multiview_v30`
- `use_physical_space_temporal_loss_v29`
- variable-view training

Expected clean val_MPJPE on the mixed WebBridge H36M+MPI manifest: ** 28–40 mm** (local smoke) / **< 25 mm** (full A800 run).

### Targets for v33

| Metric | Baseline (v30/v31/v32) | v33 target |
|--------|------------------------|------------|
| Clean val MPJPE | ~25–40 mm | −3 mm relative (e.g. 22–37 mm) |
| MPJPE@k=2 (variable views) | Baseline + X mm | ≥ 5% relative improvement |
| MPJPE@k=4 | Baseline + Y mm | ≥ 5% relative improvement |
| Per-joint error (wrist/ankle) | baseline | ≥ 5% reduction |
| Inference FPS | unchanged | unchanged (teacher is frozen and can be cached) |

**Success criterion:** v33 with `v33_distill_alpha=0.1` and `v33_distill_beta=0.05` outperforms the no-distillation run on both clean MPJPE and the variable-view curve (k = 2..14) without regressing runtime.

### Evaluation protocol

1. Run the full training with and without `--use_monocular_distillation_v33` (identical seed/config otherwise).
2. Report:
   - `val_MPJPE` (mm)
   - `val_PA_MPJPE` if available
   - Variable-view curve `MPJPE@k` for k = 2..14 using the existing variable-view mask logic.
   - Robustness matrix: rot/trans/focal/pp perturbations from the existing calibration curriculum.
3. Optional: run `experiments/benchmark_runtime.py` to confirm no FPS regression.

## 5. Risks / unknowns

| Risk | Mitigation |
|------|------------|
| Single-view self-teacher is too weak to add signal | Sweep teacher types; treat external CLIFF/PARE as follow-up. |
| Monocular predictions are in a different coordinate frame | Always Procrustes-align to world-frame ground truth before distillation loss. |
| Extra forward pass increases memory/time | Freeze teacher; cache teacher outputs per epoch if possible. |
| Distillation loss overpowers ground-trune MSE | Use small `α/β/γ` and warm up the weight linearly over 3 epochs. |
| No RGB data for Option B external models | Document Option A as the default; Option B gated by checkpoint availability. |
| Variable-view teacher mismatch (teacher sees V=1, student sees V>1) | Use per-view teacher outputs and align in student frame; do not require teacher to know the full rig. |

## 6. Implementation checklist

1. Add `MonocularDistillationTeacher` and `MonocularDistillationLoss` modules.
2. Add `use_monocular_distillation_v33` and related flags to `omniview_fusion_v5.py`.
3. Add argument parsing and loss wiring to `train_omniview_fusion_v5_webbridge_multi.py`.
4. Add smoke test `tests/test_monocular_distillation_v33.py`.
5. Run CPU smoke, then a small RTX 4090 run (d=32, 2–3 epochs).
6. If smoke is promising, add to `scripts/launch_v33_a800_queue.py` for the A800 queue.

## 7. Related files

- `motionflow_mv/fusion/omniview_fusion_v5.py`
- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
- `motionflow_mv/models/distilled_student_principal_point_model.py`
- `experiments/train_distilled_student_pp_mpiinf3dhp.py`
- `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`
- `docs/swarm_iter_next/distilled_student_kd_proposal.md` (prior multi-view KD work)
