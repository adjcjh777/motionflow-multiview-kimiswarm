# v51 Test-Time Self-Evolution Refiner (TTSER)

**Focus area:** test_time_adaptation  
**Module:** `TestTimeSelfEvolutionRefinerV51` → `motionflow_mv/fusion/test_time_self_evolution_v51.py`

## Purpose

v51 closes the self-evolution loop at inference time. v50 SEFH learns a trained critic that refines per-view reliability and per-joint uncertainty; v51 is a label-free, per-sequence optimizer that adapts the same quantities on the fly using only geometric self-consistency. It targets drift in sparse-view, temporal, and cross-domain conditions without touching base-model weights or requiring extra annotations.

## Architecture description

For each input clip, allocate a small per-sequence adaptation buffer:

- per-view reliability offset `ρ_v`,
- per-joint log-uncertainty `log σ_j`.

Both are initialized from the v50 SEFH outputs when `v51_tta_use_sefh_init=True`; otherwise they start at neutral values. From the current v50 pose estimate, compute four residuals: reprojection, temporal jump (when clip length > 1), epipolar line distance, and bone-length deviation from a learnt skeleton prior. A 2-layer MLP (`v51_tta_hidden=32`) predicts the update directions `Δρ, Δlogσ`. Run `v51_tta_num_steps=3` Adam steps on the buffer only, then feed the refined `sigmoid(ρ_v)` and `exp(log σ_j)` back into the final triangulation/aggregation step. The module is identity-at-init: with `v51_tta_num_steps=0` the pipeline reduces to v50 exactly.

## New config flags with defaults

| Flag | Type | Default | Description |
|---|---|---|---|
| `use_v51_test_time_self_evolution_refiner` | bool | `False` | Master switch. |
| `v51_tta_num_steps` | int | `3` | Gradient steps per clip at test time. |
| `v51_tta_lr` | float | `1e-3` | Adam LR for the per-sequence buffer. |
| `v51_tta_hidden` | int | `32` | Hidden dim of the update MLP. |
| `v51_tta_reproj_weight` | float | `1.0` | Weight of the reprojection-consistency term. |
| `v51_tta_temporal_weight` | float | `0.5` | Weight of the temporal-smoothness term. |
| `v51_tta_bone_weight` | float | `0.1` | Weight of the bone-length prior term. |
| `v51_tta_entropy_weight` | float | `0.01` | Entropy regularization on reliability. |
| `v51_tta_min_view_rel` | float | `0.05` | Floor on per-view reliability. |
| `v51_tta_max_view_rel` | float | `1.0` | Ceiling on per-view reliability. |
| `v51_tta_use_sefh_init` | bool | `True` | Initialize buffer from v50 SEFH outputs. |

## Loss term

The test-time loss is fully self-supervised:

```
L_TTA = w_reproj · Σ_v sigmoid(ρ_v) · r²_reproj[v]
      + w_temp · Σ_j exp(−σ_j) · r²_temp[j]
      + w_bone · L_bone(σ)
      + w_ent · L_entropy(ρ)
```

No labeled 3-D pose is used; the loss depends only on 2-D inputs, cameras, and the current pose estimate. The entropy term prevents collapse to uniform reliability, and the temporal term is disabled for single-frame inputs.

## Evaluation metric

- `MPJPE@k` for `k = 2, 3, 4, full` on H36M / MPI / 3DPW actual.
- `ΔMPJPE = MPJPE(TTSER) − MPJPE(v50)` per domain and view count.
- `Spearman(refined_reliability, reprojection_residual)` target > 0.3.
- Inference overhead: wall-clock ms per clip, target < 20 % increase over v50.

## Expected MPJPE impact

- **3DPW actual (cross-domain):** MPJPE@2 −3 to −5 mm, MPJPE@3 −2 to −4 mm. Gains come from online correction of appearance/calibration drift.
- **H36M / MPI in-domain:** MPJPE@2 −1 to −2 mm; full-view ±0.3 mm.
- **Sparse-view robustness:** lower variance in MPJPE@2/3 through better reliability ranking of dropped views.

## Main risk

**Over-adaptation to test-sequence noise.** On very short clips or when the initial v50 pose is already poor, optimizing the buffer can amplify errors. Mitigation: clamp reliability to `[v51_tta_min_view_rel, v51_tta_max_view_rel]`, cap gradient steps at 3, initialize from v50 SEFH, and monitor the reprojection loss; if it increases, fall back to the v50 output.
