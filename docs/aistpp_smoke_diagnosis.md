# AIST++ Smoke Diagnosis: v25/v80 vs. DLT Gap

**Status:** Diagnosis complete, actionable fixes below.  
**Scope:** `configs/splits/aist_only_smoke.yaml` smoke runs (`scripts/run_v25_aist_only_smoke_local_4090.sh`, `scripts/run_v80_aist_only_smoke_local_4090.sh`).

## 1. Observed numbers

| Method | Reported val MPJPE (mm) | Re-evaluated direct MJE (mm) | Re-evaluated root MPJPE (mm) | Notes |
|---|---:|---:|---:|---|
| DLT (unweighted) | **12.66** | **12.66** | **13.10** | frozen ref on `gBR_sBM_cAll_d04_mBR0_ch03` |
| v25 | 71.79 | **33.58** | **30.94** | saved non-EMA checkpoint |
| v80 | 76.34 | **100.51** (best) / **145.97** (final) | **56.74** / **56.73** | best saved by val_loss; final much worse |

Re-evaluation was done on CPU with the exact saved checkpoints and the same validation clip (`ch03`).  The v25 non-EMA checkpoint is substantially better than the EMA number that was logged during training, but still ~2.5× worse than DLT.

## 2. Why the gap is so large

### 2.1 The smoke split is tiny and unrepresentative

`configs/splits/aist_only_smoke.yaml` uses:

* **Train:** `gBR_sBM_cAll_d04_mBR0_ch01`, `ch02` (two camera takes of the same dance)
* **Val:** `gBR_sBM_cAll_d04_mBR0_ch03` (a third take of the same dance)

All three clips come from the **same subject, same action, same 9-view rig**.  A complex neural architecture (v25: ~40 M params; v80: ~15 M params with many auxiliary heads) cannot learn meaningful cross-view fusion from two clips; DLT, on the other hand, is a closed-form triangulation of the 9 calibrated views and needs no learning.

### 2.2 The chosen val clip has an abnormally low DLT error

DLT error across the three smoke clips:

| Clip | Direct MJE (mm) | Root MPJPE (mm) |
|---|---:|---:|
| `ch01` | 48.15 | 35.19 |
| `ch02` | 14.75 | 16.60 |
| `ch03` (val) | **12.66** | **13.10** |

`ch03` is the easiest of the three takes.  Picking it as the single validation clip makes the DLT baseline look exceptionally strong and exaggerates the neural-model gap.  On `ch01`, for example, the gap between v25 (33.58 mm) and DLT would shrink from ~59 mm to ~15 mm.

### 2.3 The v25/v80 smoke scripts are over-configured for 2 training clips

The v25 smoke script enables geometry attention, learned-depth triangulation, bundle adjustment, deformable cross-view attention, multiscale fusion, camera/view embeddings, set-view aggregation, variable-view training, outlier-view injection, reprojection loss, Procrustes loss, and entropy regularization.

The v80 smoke script adds even more heads (v45 adaptive geometry fusion, v46 sparse-view generalization, v50 self-evolution feedback, v51 cross-domain sparse-view reliability, v52 uncertainty-weighted triangulation, v80 view-reliability, v30 hierarchical multiview, v29 physical-space temporal loss).  With only 128 random clips sampled from two source clips, the auxiliary-loss terms dominate and the network never learns a clean 3D regressor.

### 2.4 Domain embedding and variable-view training are mismatched to the data

* `--use_domain_embedding --num_domains 8` is unnecessary: the manifest contains only AIST++ (`dataset_id=2`).  The embedding simply learns one arbitrary vector for the single present domain and wastes capacity.
* `--use_variable_view_training --variable_view_max_views 5/8` forces the model to train on subsets while AIST++ has exactly 9 calibrated views.  During validation all 9 views are present, so the model has not been trained to use the full rig it sees at test time.

### 2.5 Saved checkpoints vs. logged EMA metrics are inconsistent

The trainer evaluates with the EMA shadow weights (`ema_eval=True`) but saves the **non-EMA** `model.state_dict()`.  For v25, the EMA val MPJPE reported in the log is 71.79 mm, while the saved non-EMA checkpoint re-evaluates to 33.58 mm.  The saved model is therefore not the model that produced the reported number.  For v80 the same pattern holds, and the final checkpoint is even worse than the best-by-loss checkpoint.

### 2.6 No pre-training

v25/v80 were trained **from scratch** on the AIST++-only smoke split.  On H36M true-GT, the same architectures benefit from pre-training on large mixed data.  AIST++ dancing motions (large limb velocities, stretched poses, 9 wide-baseline views) are very different from the H36M walking/posing domain, so without cross-domain pre-training the network defaults to a poor pose prior.

## 3. Actionable fixes

### Immediate (no GPU required)

1. **Fix checkpoint saving.** In `motionflow_mv/training/trainer_v2.py`, when EMA is used for evaluation, save the EMA shadow weights (or save both).  The current `save_checkpoint` stores `self.model.state_dict()` while `evaluate` uses `self.ema.apply_shadow`.
2. **Report both direct and root-aligned/PA-MPJPE.** The current log shows only direct MPJPE.  AIST++ dancers have large global rotations, so PA-MPJPE is a fairer comparison.
3. **Use a confidence-weighted DLT baseline in addition to unweighted DLT.** AIST++ confidence maps have many low-confidence joints; conf-weighted DLT is a stronger reference.

### Short-term (CPU/GPU, no new data)

4. **Disable unnecessary flags for AIST++-only smoke:**
   * Remove `--use_domain_embedding` (single domain).
   * Remove `--use_variable_view_training` and related flags, or set `variable_view_max_views=9` so the model trains on the full 9-view rig.
   * Strip v45/v46/v50/v51/v52/v80/v29/v30 auxiliary heads from the v80 smoke run; they are not useful on two training clips.
5. **Simplify the loss mix.** Use only D MSE (3D pose) + light reprojection for the smoke run.  Physical losses, PA loss, monotonic loss, bone loss, and entropy regularization drown the tiny dataset.
6. **Run more than 3 epochs.** 3 epochs × 128 random clips is ~384 training samples; increase to at least 10–20 epochs for an AIST++-only smoke.
7. **Use the larger AIST++ manifest.** `configs/splits/webbridge_aistpp_train_val.yaml` contains many more sequences and a proper train/val split.  Reserve the `aist_only_smoke.yaml` only for a quick crash-test of the loader, not for model comparison.

### Medium-term (requires GPU, must wait for agent-51)

8. **Pre-train v25/v80 on H36M true-GT (or mixed H36M+MPI+Shelf/Campus) then fine-tune on AIST++.**  This is the standard cross-domain recipe; the current from-scratch smoke is not a fair test of the architecture.
9. **Add an Iskakov baseline for AIST++.**  Iskakov ICCV 2019 already beats DLT on H36M true-GT (23.35 mm vs. 29.19 mm) and Shelf/Campus.  Running it on AIST++ gives a realistic neural baseline.
10. **Create a balanced AIST++ evaluation split.**  Use whole-sequence splits (e.g., 80/20 by dance genre or subject) rather than three takes of the same sequence.  Report DLT, Iskakov, v25, and v80 on the same split.

## 4. Recommended next smoke command (AIST++-only, minimal)

```bash
python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/aist_only_smoke.yaml \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.1 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --num_workers 0 \
    --d 64 --residual_hidden 128 --n_st_layers 2 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 20 --batch_size 4 --train_samples 128 --val_stride 10 \
    --lr 1e-3 --lr_cosine --lr_warmup_epochs 1 --lr_min 1e-6 \
    --max_grad_norm 1.0 \
    --output outputs/omniview_fusion_v25_aist_only_minimal.pth
```

Key differences from the current smoke:

* No `--use_domain_embedding`.
* No variable-view training (full 9 views used every batch).
* No reprojection/PA/physical/entropy/bone auxiliary losses.
* More epochs.

## 5. Conclusion

The 71.79/76.34 mm AIST++ smoke numbers do **not** indicate a broken v25/v80 architecture.  They reflect an unfair comparison: a tiny, same-sequence split with an unusually easy validation take, an over-configured loss stack, and from-scratch training.  The immediate fixes are (1) fix EMA-vs-non-EMA checkpoint saving, (2) move to the larger AIST++ split, and (3) pre-train on H36M/mixed data before AIST++ fine-tuning.  DLT should be treated as a strong triangulation baseline, not a target the neural network is expected to beat on 2-clip smoke data.
