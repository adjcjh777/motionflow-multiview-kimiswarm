# v44/v45 A800 GPU Queue Plan

This document proposes a redesigned A800-D GPU queue for the v44 architecture decision and the first v45 explorations. It is meant to replace the long, unconditional `launch_v33_a800_queue.py` list with a smaller, decision-gated queue that only launches experiments that can still change the v44/v45 direction.

## Context

Per `docs/results_snapshot_2026_08_09.md` and `docs/v44_decision_plan.md`:

- The strongest known baseline is **v25 geometry fusion** (17.17 mm on A800).
- The v31-v34 complex stacks are 26-37 mm, i.e. far behind v25.
- Six decision runs are already queued on A800 in `scripts/launch_v33_a800_queue.py`.
- v44 will most likely be a v25-based architecture; v45 will extend that architecture with targeted additions (adaptive triangulation, sparse-view robustness, temporal aggregation).

## Design principles

1. **Gate everything on the v44 decision.** Do not launch new runs until the Phase 0 decision gate has reported epoch-1 validation MPJPE.
2. **Prioritize v25-based branches.** Given current evidence, the queue should prepare the v25-based v44 branch first and only keep the complex-stack branch as a fallback.
3. **Drop low-priority v31-v34 ablations.** Many v31-v34 runs in `launch_v33_a800_queue.py` are no longer decision-critical once v25 has shown a ~9 mm advantage. They are deprioritized below v44/v45 runs.
4. **Use high-capacity, strongly regularised configs.** All full runs use `d=128`, `n_st_layers=3`, `batch_size=16`, `clip_len=13`, and `weight_decay=1e-4` to give complex components a fair chance.
5. **Respect GPU memory.** A d=128 run needs roughly one A800 80 GB GPU. Co-location is only allowed for d=64 smoke or small runs.

## Phase 0: v44 decision gate (already queued)

Wait for these runs from `scripts/launch_v33_a800_queue.py` before launching any v44/v45 runs.

| Priority | Run | Question it answers |
|---|---|---|
| 1 | `v25_geometry_fusion_all_train_baseline` | How strong is plain v25 on the full mixed manifest? |
| 2 | `v25_geometry_fusion_all_train_plus_physical_domain` | Do v40 physical loss + v41 domain weights help v25? |
| 3 | `v42_v36_physical_domain_no_v37` | Does the v36+physical+domain stack work on expanded data? |
| 4 | `v43_adaptive_node_residual_on_v42` | Does the v43 per-node adaptive residual help? |
| 5 | `v43_adaptive_node_residual_scaled` | Does capacity help the v43 stack? |
| 6 | `v43_adaptive_node_residual_all_train` | Does the full WebBridge mixed manifest help the v43 stack? |

### Decision tree (from `docs/v43_decision_criteria.md` and `docs/v44_decision_plan.md`)

- **Branch A (most likely):** `v25 all-train baseline` is best or within ~1 mm of best.
  - v44 = v25 + selective additions (physical/domain, SWA, longer training).
- **Branch B:** `v42` clearly beats v25.
  - v44 = v42 + stronger regularisation (dropout, weight decay, SWA).
- **Branch C:** `v43 base` beats v42 by >5% relative.
  - v44 = v43 + v44 edge-type-aware gating.
- **Branch D:** `v43 scaled` or `v43 all-train` is best.
  - v44 = scale v25 or v43 further, with stronger anti-overfitting.

## Phase 1: v44 candidate runs

These runs are launched **only after** the Phase 0 decision gate reports epoch-1 val MPJPE. Each run uses the full mixed manifest (`configs/splits/webbridge_all_train_mixed.yaml`) unless noted.

### Shared v25 base flags

```text
--use_mixed_loader --use_domain_embedding
--use_deformable_cross_view_attention_v18
--use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2
--v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment
--use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true
--use_context_visibility true --use_skeleton_residual true --use_rotation_correction true
--use_entropy_regularization true --attention_entropy_weight 0.01
--use_camera_view_embedding --use_set_view_aggregator
--use_variable_view_training --variable_view_minViews 2 --variable_view_max_views 14
--variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute
--pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0
--reproj_loss_weight 0.1 --reproj_warmup_epochs 3 --aleatoric_reproj_loss_weight 0.1
--outlier_view_prob 0.3 --outlier_view_max_views 1
--outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0
```

### Branch A: v25 wins (prepare first)

| Run | Condition | Extra flags | Capacity |
|---|---|---|---|
| `v44_v25_all_train_swa` | Always run | `--ema_decay 0.999 --early_stopping_patience 5 --early_stopping_min_delta 0.001` | `d=128 residual_hidden=256 n_st_layers=3 batch_size=16 clip_len=13 train_samples=200 epochs=20 weight_decay=1e-4` |
| `v44_v25_plus_physical_domain` | If P0 #2 beats P0 #1 | `--use_skeleton_physical_loss_v40 --v40_bone_weight 0.05 --v40_joint_limit_weight 0.01 --v40_symmetry_weight 0.02 --v40_floor_weight 0.02 --domain_loss_weights 1.0,1.5` | Same as above |
| `v44_v25_plus_learned_triangulation` | If v33 uncertainty-aware triangulation shows promise in local smoke | `--use_uncertainty_aware_triangulation_v33 --v33_uat_loss_weight 0.01` | Same as above |
| `v44_v25_longer_clip` | If temporal jitter is flagged as an issue | `--clip_len 13` (baseline is 9) | Same as above |

### Branch B: v42 wins

| Run | Condition | Extra flags | Capacity |
|---|---|---|---|
| `v44_complex_v42_strong_reg` | v42 beats v25 | v42 flags + `--weight_decay 1e-4 --v25_dropout 0.3 --v30_stochastic_depth_prob 0.2 --ema_decay 0.999` | `d=128 residual_hidden=256 n_st_layers=3 batch_size=16 clip_len=13 train_samples=200 epochs=20` |

### Branch C: v43 base wins

| Run | Condition | Extra flags | Capacity |
|---|---|---|---|
| `v44_complex_v43_edge_gate` | v43 base beats v42 by >5% | v43 flags + `--use_v44_edge_type_gate` | Same as above |

### Branch D: capacity/data wins

| Run | Condition | Extra flags | Capacity |
|---|---|---|---|
| `v44_complex_v43_scaled_full` | v43 scaled/all-train is best | v43 flags + `--d 128 --residual_hidden 256 --n_st_layers 3 --batch_size 16 --clip_len 13 --train_samples 10000 --epochs 5 --early_stopping_patience 2` | One full A800 GPU per run |

## Phase 2: v45 exploration runs

v45 experiments are launched **only after** a v44 candidate is selected and has a stable epoch-1 result. All v45 runs assume a v25-based v44 (Branch A). If v44 turns out to be a complex stack, the same flags can be prefixed onto the v43/v44 base.

| Variant | Run | Extra flags | What it tests |
|---|---|---|---|
| **v45-AGF** adaptive geometry fusion | `v45_agf_v25_uncertainty_triangulation` | `--use_uncertainty_aware_triangulation_v33 --v33_uat_loss_weight 0.01` | Learned per-view triangulation weights |
| **v45-AGF** + physical/domain | `v45_agf_v25_physical_domain` | `--use_uncertainty_aware_triangulation_v33 --v33_uat_loss_weight 0.01 --use_skeleton_physical_loss_v40 --v40_bone_weight 0.05 --v40_joint_limit_weight 0.01 --v40_symmetry_weight 0.02 --v40_floor_weight 0.02 --domain_loss_weights 1.0,1.5` | Whether AGF and physical/domain are complementary |
| **v45-SVG** sparse-view generalisation | `v45_svg_v25_strong_view_dropout` | Increase `--outlier_view_prob 0.5 --outlier_view_max_views 2` (or add new view-dropout flag when implemented) | Robustness to missing views |
| **v45-TGA** temporal geometry | `v45_tga_v25_clip13` | `--clip_len 13` | Longer temporal window |
| **v45-TGA** + temporal loss | `v45_tga_v25_temporal_loss` | `--clip_len 13 --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001 --v29_physical_loss_warmup_epochs 3` | Temporal smoothness prior |
| **v45-combined** | `v45_combined_v25` | Combine the two strongest independent v45 additions on top of v44 | Best v45 system |

## Resource allocation and scheduling

### GPU memory thresholds

| Config type | `MIN_FREE_MIB` | Runs per A800 GPU | Notes |
|---|---|---|---|
| d=64, batch=8, clip=9 | 30 000 | Up to 2 | Phase 0 #3-#6, small ablations |
| d=128, batch=16, clip=13 | 50 000 | 1 | Phase 1/2 full runs |

### tmux naming convention

```text
v44_<name>_gpu<N>
v45_<name>_gpu<N>
```

Example: `v44_v25_plus_physical_domain_gpu4`.

### Launch order

1. Poll A800-D with `nvidia-smi`.
2. Wait until **all Phase 0 decision runs** have finished epoch 1 and their val_MPJPE is recorded.
3. Apply the decision tree in Section 3 to choose Branch A/B/C/D.
4. Launch Phase 1 runs for the chosen branch only.
5. After the best v44 run is identified, launch Phase 2 v45 runs.

## Proposed launch script changes

Rather than extending `launch_v33_a800_queue.py` unconditionally, create a new `scripts/launch_v44_v45_a800_queue.py` that:

- Removes or comments out the low-priority v31-v34 ablations.
- Keeps only the six Phase 0 decision runs.
- Adds the Phase 1/2 runs above as **separate lists** with a `branch` tag.
- Implements the decision tree as a function:

```python
def eligible_runs(phase0_results: dict[str, float]) -> list[tuple[str, str, str]]:
    v25 = phase0_results["v25_geometry_fusion_all_train_baseline"]
    v25pd = phase0_results["v25_geometry_fusion_all_train_plus_physical_domain"]
    v42 = phase0_results["v42_v36_physical_domain_no_v37"]
    v43 = phase0_results["v43_adaptive_node_residual_on_v42"]

    runs = ["v44_v25_all_train_swa"]
    if v25pd < v25:
        runs.append("v44_v25_plus_physical_domain")
    if v42 < v25 - 1.0:
        runs.append("v44_complex_v42_strong_reg")
    if v43 < v42 * 0.95:
        runs.append("v44_complex_v43_edge_gate")
    # ... add v45 runs once v44 is selected
    return runs
```

This keeps the queue compact and prevents A800 GPUs from being consumed by experiments that are no longer decision-critical.

## Risks and fallbacks

| Risk | Mitigation |
|---|---|
| v25 overfits after epoch 1 | Use `early_stopping_patience=2`, `weight_decay=1e-4`, and EMA in all v44/v45 runs. |
| d=128 runs OOM | Fall back to `d=64, batch_size=8, clip_len=9` and increase `train_samples` to compensate. |
| v44 edge-type gate requires v36 | Keep it in Branch C only; do not force it onto the v25 base. |
| Phase 0 runs are still running on A800 | This plan explicitly waits for them; do not add competing large runs before they finish. |
| New v45 modules are not yet implemented | v45-AGF can use existing `--use_uncertainty_aware_triangulation_v33` as a proxy; v45-SVG/TGA are gated behind implementation. |

## Next steps

1. Wait for the six Phase 0 A800 runs to report epoch-1 val_MPJPE.
2. Update `docs/results_snapshot_2026_08_09.md` with the new numbers.
3. Choose Branch A/B/C/D and create `scripts/launch_v44_v45_a800_queue.py` with the corresponding run list.
4. Run a local RTX 4090 smoke of the chosen v44 config before any full A800 launch.
5. Launch Phase 1, then Phase 2.
