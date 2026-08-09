# Agent-18 Report: Mapping Qwen3.8 Self-Evolution to the v46 Design-Train-Evaluate Loop

**Owner:** Agent-18  
**Task:** ANALYZE — Map Qwen3.8 self-evolution concepts to our design-train-evaluate loop.  
**Output:** `docs/swarm_iter23/reports/agent18_qwen_selfevolution.md`  
**Tracking issue:** #160

---

## 1. Qwen3.8 self-evolution concepts at a glance

Qwen3.8-style self-evolution can be decomposed into four repeating mechanisms:

| Mechanism | What it means in LLMs | MotionFlow-MultiView analogue |
|-----------|----------------------|-------------------------------|
| **Self-critique** | Model scores/flags its own outputs via a critique head or reward model. | `v37` self-critique view reliability learns per-(view,joint) reliability from reprojection residuals. |
| **Iterative refinement** | Output is fed back into the model for successive improvement. | `v36` UGIGR iteratively refines pose via uncertainty-gated graph refinement. |
| **Curriculum / adaptive sampling** | Difficulty of training examples is progressively increased. | `v46` sparse-view curriculum gradually raises view-dropout probability. |
| **Selection / branch decision** | Best checkpoint/branch is chosen by an internal or external evaluator. | `v44` decision plan selects the next architecture branch based on A800 val_MPJPE. |

This report maps each concept to our concrete **design-train-evaluate** loop and identifies where v46-SVG can plug in.

---

## 2. Design phase: self-critique as an architecture signal

In Qwen3.8, self-critique starts at **design time**: the model learns to predict whether a token/answer is likely to be correct. In our codebase, this is already instantiated in `v37` and `v39`:

- **`v37` reliability head** predicts a per-(view, joint) reliability score from reprojection residuals.
- **`v39` closes the loop** by using the reliability score to gate uncertainty in `v36` UGIGR.
- **`v45-AGF`** reuses the same idea for per-view/per-joint triangulation weights.

### v46-SVG mapping

For sparse-view generalization, the design should include a **per-view reliability head** that explicitly handles missing views:

```text
Input features (B, T, V, J, C)
    |
    ▼
Per-view reliability head
    |
    ├── Predict r_v ∈ (0,1) for each available view
    ├── Mask dropped views to r_v = 0
    └── Feed r_v into weighted DLT / geometry fusion
```

This is the same self-critique pattern: the model assigns a confidence to each view before fusing them.

**Design recommendation for v46:**
- Reuse the `v45-AGF` reliability weighting path rather than inventing a new graph module.
- Add a lightweight MLP that takes the same per-view feature and outputs a dropout-aware reliability score.
- Keep the mask application outside the MLP so the head only learns from available views.

---

## 3. Train phase: iterative refinement + curriculum as a training loop

### 3.1 Iterative refinement in training

Qwen3.8 refines outputs by passing them through the model multiple times. In our pipeline:

- **`v36` UGIGR** already does this: pose estimates are refined across graph iterations.
- **`v39` reliability-coupled refinement** uses the self-critique score to decide how much refinement to apply per node.

### 3.2 Curriculum as adaptive augmentation

Qwen3.8 often uses curriculum learning to progressively increase task difficulty. In v46, this maps directly to view-dropout augmentation:

```
Epoch 0-1:  p_drop = 0.0   (full views, learn baseline)
Epoch 2-3:  p_drop = 0.1   (slightly sparse)
Epoch 4-5:  p_drop = 0.2   (moderately sparse)
Epoch 6+:   p_drop = 0.3   (target sparsity)
```

This mirrors the `v46_svg_use_curriculum` flag proposed in `docs/proposals/v46_sparse_view_generalization.md`.

### 3.3 Training-loop integration

The training loop should be modified as follows:

```python
# Pseudo-code for the v46 training loop
features = model.extract_features(images)          # (B, T, V, J, C)
view_mask, dropout_prob = drop_views(...)          # v46 augmentation
reliability = svg_module(features, view_mask)      # self-critique
output = geometry_fusion(features, reliability)    # iterative refinement
loss = pose_loss(output, gt) + reliability_loss(...)
```

Key constraints from the proposal:
- `min_views >= 2` to keep triangulation valid.
- Apply dropout **inside the training loop**, not in the loader, to preserve data determinism.
- Make dropout probability curriculum-aware.

---

## 4. Evaluate phase: self-selection and branch decision

### 4.1 Self-selection

Qwen3.8 selects the best answer from multiple candidates using a reward model. Our equivalent is **validation-based checkpoint selection**:

- `v25` full run selected by best `val_MPJPE` (17.17 mm on A800).
- `v42`/`v43` are compared against `v25` in the `v44` decision plan.
- `v46` smoke selects whether sparse-view training is worth a full A800 run.

### 4.2 Metrics as reward signals

The Qwen reward model can be seen as a scalar that guides which branch to keep. For v46, the reward signal is:

| Metric | Role |
|--------|------|
| `val_MPJPE` | Overall pose quality at full views |
| `MPJPE@k` for k=2,3,4 | Sparse-view generalization (the new reward) |
| `Dropout robustness curve` | Monitors degradation as views drop |

### 4.3 v46 branch decision rule

Following the `v44` decision plan style, the v46 self-evolution rule can be written as:

```
if v46_smoke_val_MPJPE < 80 mm and MPJPE@2/3 improves over v45:
    promote v46 to full A800 queue
else:
    keep v45-AGF and revisit dropout/reliability design
```

---

## 5. Closing the loop: from evaluation back to design

A true self-evolving system feeds evaluation results back into the next design. For the v46 swarm, this means:

1. **Smoke results** → decide whether to keep the current `SparseViewGeneralizationV46` API or add/remove components.
2. **A800 MPJPE@k** → decide if the reliability head needs more capacity, curriculum schedule tuning, or integration with `v45-AGF` weights.
3. **v47 combined architecture** → if v46 succeeds, combine it with temporal aggregation (Agent-19 task).

This is the same iterative improvement loop as Qwen3.8: **design → train → evaluate → critique → redesign**.

---

## 6. Concrete recommendations for v46-SVG

1. **Reuse self-critique weights from v45/v37.** Do not add a new heavy transformer; the v46 module should be a small reliability head on top of existing features.
2. **Implement curriculum dropout as a training-loop augmentation.** Start at `p_drop=0.0` and ramp to the target probability over the first few epochs.
3. **Use `MPJPE@k` as the reward/selection metric.** This is the v46-specific success signal, analogous to a reward model score.
4. **Wire the reliability mask into existing weighted triangulation.** Zeroing out dropped views is the geometric equivalent of masking invalid tokens in an LLM.
5. **Treat v46 as an augmentation, not a replacement.** Geometry fusion (v25) remains the foundation; v46 makes it robust to missing views.

---

## 7. Summary

Qwen3.8 self-evolution maps cleanly onto our v46 workflow:

- **Self-critique** → per-view reliability head (v37/v45 reuse).
- **Iterative refinement** → UGIGR-style graph refinement and weighted triangulation.
- **Curriculum** → progressive view-dropout augmentation.
- **Selection/branching** → validation `MPJPE@k` and the v46 → v47 promotion decision.

The v46-SVG design should therefore be understood as **a self-evolving augmentation loop**: the model learns to critique which views are reliable under sparse conditions, and the training curriculum progressively exposes it to harder sparse-view configurations.
