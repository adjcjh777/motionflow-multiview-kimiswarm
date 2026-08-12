# v52 Self-Supervised Multi-View Pre-Training Head — Risk Report

**Module:** `self_supervised_multiview_pretraining_v52`  
**Date:** 2026-08-09  
**Agent:** design-swarm-agent-v52

## Risk 1: Auxiliary loss dominates early training

**Description.**  The v52 reconstruction losses can overwhelm the supervised 3D pose objective in the first epochs, causing the main MPJPE to rise before the module has learned useful features.

**Mitigation.**
- Start with small weights (`v52_loss_2d_weight=0.05`, `v52_loss_geo_weight=0.05`, `v52_loss_cont_weight=0.005`) and warm them up linearly over `v52_warmup_epochs`.
- Monitor the ratio `L_v52 / L_total` in tensorboard; cap it at 0.20 by dynamically down-weighting if it exceeds the cap.
- Run the first smoke with the module enabled but weights set to zero to verify identity-at-init does not perturb the baseline.

## Risk 2: Masking too aggressively removes the views/joints needed for accurate triangulation

**Description.**  Random masking can leave a joint with fewer than two visible views, making DLT-based triangulation unstable and producing noisy 3D targets for the geometry loss.

**Mitigation.**
- Enforce `v52_min_visible_views >= 2` per joint per time step using the existing `view_mask` helper.
- Reject sampled masks that would drop a joint below `min_visible_views`; resample up to 5 times before falling back to a conservative mask.
- In the geometry loss, weight each joint by `min_v conf / reprojection_error` so unstable samples contribute less.

## Risk 3: Contrastive loss is slow / memory heavy

**Description.**  A global contrastive loss over all `(view, joint, time)` tokens has `O((B·T·V·J)²)` memory cost, which can OOM on the A800 for long clips or large batch sizes.

**Mitigation.**
- Use a small memory bank of 512–1024 negatives instead of all negatives.
- Apply the contrastive loss only within a local temporal window (`v52_temp_cont_window`) and sample at most 256 positive pairs per batch.
- Provide a flag `v52_use_contrastive=false` so the module can be run without contrastive loss if memory is tight.

## Risk 4: Module conflicts with v50/v51 auxiliary heads

**Description.**  v50 SEFH and v51 CDSVR already add auxiliary losses.  Adding v52 may create redundant gradients or over-regularise the shared feature backbone.

**Mitigation.**
- Gate v52 so it is only active when the shared feature extractor is being trained; when fine-tuning only v50/v51 heads, set `v52_warmup_epochs=0` and `v52_loss_*_weight=0`.
- Run a 2-epoch ablation with the combinations `{baseline, +v50, +v50+v52, +v51}` to detect negative interaction before committing to a full run.

## Risk 5: Self-supervised objective does not transfer to 3D pose accuracy

**Description.**  Reconstructing 2D keypoints and soft 3D positions is a proxy task; improved reconstruction may not translate into lower MPJPE.

**Mitigation.**
- Include an explicit 3D consistency branch: after reconstruction, triangulate the masked subset and compare the result to the full-view triangulation of the *original* tokens.
- Use the existing v25/v45 triangulation module rather than a separate triangulator, ensuring the pretext task uses the same geometric inductive bias as the main task.
- Define a clear smoke acceptance rule: `val_MPJPE` must not regress by more than 1 mm compared to the v51 baseline; otherwise disable v52 and investigate.
