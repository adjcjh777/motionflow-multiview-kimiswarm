# Agent-15 Report: Mapping Qwen3.8 Self-Evolution to v47 Staged Training

**Owner:** Agent-15  
**Task:** ANALYZE — Map Qwen3.8 self-evolution to v47 staged training (freeze/unfreeze, curriculum).  
**Output:** `docs/swarm_iter24/reports/agent15_qwen_staged.md`  
**Tracking issue:** #162  

---

## 1. Summary

The v47 temporal aggregation head is not merely an extra layer on top of v46; it is the next iteration of the same self-evolving loop that already drives MotionFlow-MultiView. This report maps the Qwen3.8 self-evolution principles to the **staged training recipe** proposed for v47, with concrete freeze/unfreeze and curriculum schedules. The core claim is:

> v47 should be trained like a self-evolving system: first let the new temporal head learn on frozen, stable geometry; then open the loop and let the whole model co-evolve under a progressively harder sparse-view curriculum.

---

## 2. Qwen3.8 self-evolution concepts (recap)

From `docs/qwen38_selfevolution_mapping.md` and `docs/swarm_iter23/reports/agent18_qwen_selfevolution.md`, the four mechanisms most relevant to v47 are:

| Qwen3.8 mechanism | Meaning | MotionFlow-MultiView analogue |
|---|---|---|
| **Self-critique** | Model scores its own outputs. | v37/v39 per-(view,joint) reliability; v45-AGF adaptive weights; v46 per-view reliability head. |
| **Iterative refinement** | Outputs are fed back for successive improvement. | v36 UGIGR iterative graph refinement; v45-TGA temporal geometry aggregation; v47 temporal pose refinement. |
| **Curriculum / adaptive sampling** | Training difficulty is progressively increased. | v46 view-dropout curriculum; v47 staged unfreeze + temporal-window expansion. |
| **Selection / branch decision** | Best candidate chosen by a reward/validator. | v44 decision plan; v46/v47 `MPJPE@k` as reward signal. |

---

## 3. Mapping to v47 staged training

### 3.1 Self-critique → view-count conditioning in the temporal head

The v47 `TemporalAggregationV47` module receives the per-frame triangulated pose `P_t` and a view mask `view_mask`. Each token is concatenated with `log(n_views_t)`, so the head knows how much to trust each frame. This is the temporal analogue of v46's per-view reliability head:

```text
v46: per-view reliability r_v  →  down-weight unreliable views before triangulation.
v47: per-frame view-count log(n_views_t)  →  discount under-constrained frames before temporal fusion.
```

**Training implication:** during the first stage, the temporal head learns to interpret these self-critique signals while the underlying reliability/geometry weights remain fixed.

### 3.2 Iterative refinement → temporal aggregation as a residual loop

v47 does not replace the per-frame v46 pipeline; it refines its output:

```
P_t^(0) = v46(P_t | sparse views)
P_t^(l+1) = P_t^(l) + g · ΔP_t^(l)      # g initialised to 0
```

This mirrors Qwen's *action → feedback → retry* cycle, where each temporal layer re-examines the trajectory conditioned on neighboring frames. The **residual gate** `g` is the safety brake that lets the loop warm up from identity, exactly like the v36 source gate.

### 3.3 Curriculum → staged view dropout + staged unfreeze

Qwen's curriculum learning maps directly to a two-axis schedule in v47:

| Axis | Stage 1 (head only) | Stage 2 (end-to-end) |
|---|---|---|
| **Frozen modules** | v25/v45/v46 frozen | all unfrozen |
| **View-dropout probability** | 0.0 → 0.1 (curriculum ramp) | 0.1 → 0.3 (target) |
| **Temporal window** | local 7-frame window | optional full-clip window |
| **Loss weight on temporal smoothness** | 0.0 → 0.005 | 0.005 → 0.01 |

This is the v47-specific instantiation of Qwen's *progressive difficulty increase*.

### 3.4 Selection / reward → `MPJPE@k` as the universal reward

The reward that decides whether v47 is promoted is the same `MPJPE@k` metric used to gate v46:

```
if MPJPE@2/3(v47) < 0.95 * MPJPE@2/3(v46) and MPJPE@full not worse:
    promote v47 to full A800 run
else:
    revisit temporal head capacity or freeze duration
```

This closes the self-evolution loop: design (v47 head) → train (staged recipe) → evaluate (`MPJPE@k`) → select (promote or redesign).

---

## 4. Concrete v47 staged training recipe

### 4.1 Recommended 3-stage schedule

The following schedule is designed to be minimal, safe, and consistent with the existing trainer (`experiments/train_omniview_fusion_v5_webbridge_multi.py`), which already supports `warm_start_freeze_epochs`, `v46_svg_use_curriculum`, and variable-view curricula.

#### Stage 0 — Warm-start from a v46 checkpoint

- Load a trained v46-SVG checkpoint.
- Freeze all v25/v45/v46 parameters by default (reuse the existing `freeze_old_params` helper, extended to recognise the v46 prefix).
- Keep `v47_temporal_*` parameters trainable.

#### Stage 1 — Temporal head warm-up (1–2 epochs)

- Keep v25/v45/v46 **frozen**.
- Set `v46_svg_view_dropout_prob = 0.0` (or use the curriculum ramp starting at 0).
- Use a **local 7-frame temporal window** (`v47_temporal_window = 7`).
- Optimise only the temporal head with the base 3-D pose loss plus a small temporal smoothness term:

```python
loss = L_pose(P_t, P_gt) + v47_temporal_loss_weight * mean(|P_t - P_{t-1}|)
```

- Goal: head learns to smooth trajectories without destabilising the strong v46 per-frame estimates.

#### Stage 2 — End-to-end fine-tuning (remaining epochs)

- **Unfreeze all** parameters (`unfreeze_all`).
- Ramp `v46_svg_view_dropout_prob` to its target (e.g., 0.3) using the existing v46 curriculum logic over the first half of training.
- Optionally switch from local window to full-clip attention (`v47_temporal_window = None`) after 1 epoch, if memory allows.
- Increase `v47_temporal_loss_weight` to its target (e.g., 0.01).
- Goal: the whole model adapts to sparse views and temporal coherence jointly.

### 4.2 Pseudocode for the training loop

```python
# Already present in trainer:
#   - args.warm_start_freeze_epochs
#   - freeze_old_params / unfreeze_all
#   - v46_svg_use_curriculum and progress-based dropout ramp

if args.use_v47_temporal_aggregation:
    # Stage 1: head-only warm-up
    if epoch < args.v47_head_freeze_epochs:
        freeze_v25_v45_v46(model)
        effective_dropout = 0.0
        temporal_window = args.v47_temporal_window  # e.g. 7
        temporal_loss_weight = 0.0
    else:
        unfreeze_all(model)
        effective_dropout = ramp_v46_dropout(epoch, total_epochs)
        temporal_window = None if args.v47_temporal_full_clip_after_warmup else args.v47_temporal_window
        temporal_loss_weight = args.v47_temporal_loss_weight

    # Forward pass includes v47 head
    output = model(..., view_mask=view_mask)

    # Smoothness loss only after warm-up
    if temporal_loss_weight > 0:
        loss += temporal_loss_weight * temporal_smoothness_loss(output)
```

### 4.3 New flags to add to the trainer

| Flag | Type | Default | Role |
|---|---|---|---|
| `v47_head_freeze_epochs` | int | 1 | Epochs to freeze v25/v45/v46 and train only the temporal head. |
| `v47_temporal_full_clip_after_warmup` | bool | False | Switch from local window to full-clip attention after head warm-up. |
| `v47_temporal_loss_weight_start` | float | 0.0 | Initial temporal smoothness loss weight during head warm-up. |
| `v47_temporal_loss_weight` | float | 0.01 | Target temporal smoothness loss weight after warm-up. |
| `v47_curriculum_window` | bool | True | Expand temporal window as part of the curriculum. |

---

## 5. Mapping Qwen principles to v47 risks and mitigations

| Qwen principle | v47 risk | Mitigation in staged recipe |
|---|---|---|
| **Stable reward** | Temporal over-smoothing on fast motion | Residual gate `g` initialised to 0; local window first; smoothness loss ramped. |
| **Stable distribution** | Sparse-view dropout destabilises the frozen base | Freeze v25/v45/v46 while the head learns; ramp dropout only in Stage 2. |
| **Closed-loop feedback** | Temporal head ignores view-count signal | Concatenate `log(n_views_t)` to every token; evaluate `MPJPE@k` per sparsity. |
| **Selection/branching** | v47 regresses at full views | Gate promotion on both sparse-view gain **and** no full-view regression. |

---

## 6. Curriculum details for v47

### 6.1 View-dropout curriculum (reuses v46)

The existing v46 curriculum in `experiments/train_omniview_fusion_v5_webbridge_multi.py` ramps dropout over the first half of training:

```python
ramp_epochs = max(1, args.epochs // 2)
progress = min(1.0, max(0.0, current_epoch / ramp_epochs))
v46_dropout_prob = args.v46_svg_view_dropout_prob * progress
```

For v47, this ramp should be **disabled during Stage 1** (head-only training) and re-enabled in Stage 2. This prevents the temporal head from having to learn on noisy sparse inputs before it understands the base trajectory.

### 6.2 Temporal-window curriculum

| Phase | Window | Rationale |
|---|---|---|
| Stage 1 | 7 frames | Local context is easier, lower memory, faster convergence. |
| Stage 2 (early) | 7–13 frames | Gradually increase receptive field. |
| Stage 2 (late) | Full clip (`None`) | Long-range temporal dependencies, if memory allows. |

### 6.3 Loss-weight curriculum

```python
# Linear ramp over the first N warmup epochs
smoothness_weight = v47_temporal_loss_weight_start + \
    (v47_temporal_loss_weight - v47_temporal_loss_weight_start) * \
    min(1.0, epoch / v47_loss_warmup_epochs)
```

Starting from 0.0 avoids v28-style physical-loss instability, where a newly added loss term dominated early training.

---

## 7. Integration with existing trainer helpers

The v47 staged recipe can reuse existing infrastructure:

- `freeze_old_params` / `unfreeze_all` in `experiments/train_omniview_fusion_v5_webbridge_multi.py` — extend `new_prefixes` to include `v46`/`temporal_aggregation_v47` logic.
- v46 curriculum dropout in `motionflow_mv/data/view_dropout_augmentation_v46.py` — pass `progress=0` during Stage 1.
- Smoothness loss warmup — mirror `v40_warmup_epochs` / `reproj_warmup_epochs`.
- Variable-view subset logic — keep `min_views=2` to preserve triangulation validity.

---

## 8. Recommended smoke-test protocol

Before committing to a full A800 run, the staged recipe should be validated on the RTX 4090 smoke config:

1. Load a v46-SVG smoke checkpoint (or train one).
2. Add the v47 head with `v47_head_freeze_epochs=1`, `v47_temporal_window=7`.
3. Run 2 epochs:
   - Epoch 1: head-only, dropout=0.0, smoothness weight=0.0.
   - Epoch 2: unfreeze, dropout=0.3, smoothness weight=0.01.
4. Check:
   - `val_MPJPE@full` does not regress vs. v46.
   - `val_MPJPE@2/3` improves vs. v46.
   - No NaN / OOM.

---

## 9. Summary

Qwen3.8 self-evolution provides the right mental model for v47 training:

- **Self-critique** is encoded by view-count conditioning in the temporal head.
- **Iterative refinement** is the residual temporal aggregation loop over v46 outputs.
- **Curriculum** is the staged combination of view-dropout ramp, temporal-window expansion, and loss-weight ramp.
- **Selection** is the `MPJPE@k`-based decision to promote v47 to a full A800 run.

The staged recipe is therefore:

1. **Warm-start from v46.**  
2. **Freeze base, train head** (1 epoch, full views, local window).  
3. **Unfreeze and co-train** under the v46 sparse-view curriculum with a gradually expanded temporal window and smoothness loss.  
4. **Promote** only if sparse-view `MPJPE@k` improves by ≥5% without full-view regression.

This keeps v47 minimal, stable, and aligned with the self-evolving design philosophy already established in v36–v46.

---

## References

- `docs/qwen38_selfevolution_mapping.md`
- `docs/swarm_iter23/reports/agent18_qwen_selfevolution.md`
- `docs/proposals/v47_combined_architecture.md`
- `docs/proposals/v46_sparse_view_generalization.md`
- `motionflow_mv/data/view_dropout_augmentation_v46.py`
- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
