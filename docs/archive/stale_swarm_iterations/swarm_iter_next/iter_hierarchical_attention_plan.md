# Iteration Plan: Hierarchical Attention for Multi-View Pose Fusion

**Date:** 2026-08-06  
**Direction lead:** hierarchical_attention  
**Anchor to beat:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` at **8.75 mm MPJPE** / **4.95 mm PA-MPJPE** on MPI-INF-3DHP S2 Seq1  
**Baseline checkpoint:** `outputs/ray_attention_temporal_crossview_residual_principal_point_robust_retrain.pth`  
**GPU status:** RTX 4090 is running the Bayesian Triangulation full run — do **not** start any GPU training now. This document is design-only.

---

## 1. Motivation

The current anchor (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`, `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py:16`) mixes **time × views** in a single flat transformer sequence:

```python
# motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py:119-129
feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
for layer in self.st_transformer:
    feat = layer(feat)
```

This is simple but wasteful: every token attends to every other token, so spatially redundant cameras and temporally distant frames are treated equally. A **hierarchical attention** backbone can:

1. **Group spatially related views** first, reducing cross-view noise before global exchange.
2. **Smooth motion over time** in a dedicated temporal stage.
3. **Inject skeleton structure** via a graph-joint attention stage, enforcing bone/symmetry/cross-view constraints.

A skeleton already exists in the repo: `motionflow_mv/fusion/ray_attention_hierarchical_view_temporal_joint_residual_principal_point_model.py` implements a three-stage hierarchy (view groups → temporal → joint graph). It is currently **not wired into training or evaluation**. Completing and training it is the fastest path to beat the 8.75 mm anchor without changing the already-proven PP correction, residual MLP, or DLT triangulation stages.

---

## 2. Architecture

### 2.1 Starting point

Reuse the anchor end-to-end but replace the flat `st_transformer` block with a **hierarchical transformer block**.

Files inherited unchanged:
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py` — PP correction, weight head, residual MLP.
- `motionflow_mv/fusion/principal_point_correction.py` — `PrincipalPointCorrection.forward`.
- `motionflow_mv/fusion/ray_attention_model.py` — `_triangulate_weighted_dlt`.
- `motionflow_mv/fusion/graph_joint_relation.py` — `GraphJointRelation`, `build_edge_index`.

The new model is:
- `motionflow_mv/fusion/ray_attention_hierarchical_view_temporal_joint_residual_principal_point_model.py`
  - class `RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint`
  - block `_HierarchicalViewTemporalJointBlock`

### 2.2 Proposed three-stage hierarchy

For feature tokens `f ∈ R^(B×T×V×J×d)` after per-frame ray embedding:

```
Stage 1 — Hierarchical View Attention
    for each block:
        a) Within-group self-attention over views.
        b) Cross-group token exchange via pooled group representations.

Stage 2 — Temporal Attention
    Self-attention over T for every (view, joint) token.

Stage 3 — Skeleton-Graph Joint Attention
    Message passing on the (view, joint) graph:
        - bone edges
        - symmetry edges
        - cross-view same-joint edges
```

### 2.3 Key equations

**Within-group view attention.** For group `g` with views `V_g`:

```
attn_g = TransformerEncoder( f_{:, :, V_g, :, :} )   # (B*T*J, |V_g|, d)
```

**Cross-group exchange.** Pool each group to a single token, run `nn.MultiheadAttention`, and broadcast back:

```
group_tok_g = Mean_{v ∈ V_g} f_{:, :, v, :, :}        # (B*T*J, d)
group_toks  = Stack_g group_tok_g                     # (B*T*J, G, d)
group_toks  = LayerNorm(group_toks + MHA(group_toks)) # (B*T*J, G, d)
f_{v ∈ V_g} ← f_{v ∈ V_g} + group_toks_g
```

**Temporal attention.**

```
f = f.permute(0,2,3,1,4).reshape(B*V*J, T, d)
f = TemporalTransformerEncoderLayer(f)
```

**Skeleton-graph joint attention.**

```
f = GraphJointRelation(d, n_views, num_layers=1)(f, edge_index, edge_type)
```

### 2.4 Grouping strategy

The current implementation uses a **contiguous split** of view IDs (`self.group_sizes` in `_HierarchicalViewTemporalJointBlock.__init__`). For a 14-view MPI-INF-3DHP rig, two groups of 7 views is plausible but arbitrary. A better, still minimal, upgrade is **geometry-aware grouping** using `motionflow_mv/fusion/camera_positional_encoding.py` (already designed for variable rigs):

1. Cluster views by camera-center angle/floor distance (K-means with `k = n_view_groups`).
2. Pass the group assignment as a mask to the within-group attention stage.
3. Keep the cross-group exchange as above.

This makes the hierarchy robust to arbitrary camera order and a natural fit for variable-view inference.

---

## 3. Code Changes Needed

No existing source/config/running experiment should be modified. The following additions are required.

### 3.1 Register the existing model

In `motionflow_mv/fusion/__init__.py` (currently only registers the flat PP model), add:

```python
from .ray_attention_hierarchical_view_temporal_joint_residual_principal_point_model import (
    RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint,
)
```

### 3.2 Add a `FusionModule` wrapper

Create `motionflow_mv/fusion/ray_attention_hierarchical_view_temporal_joint_residual_principal_point_module.py`:

```python
class RayAttentionHierarchicalViewTemporalJointResidualPrincipalPointFusionModule(FusionModule):
    name = "ray_attention_hierarchical_view_temporal_joint_residual_principal_point"
    ...
```

Mirror `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_module.py` and call `register_ray_attention_hierarchical_view_temporal_joint_residual_principal_point_fusion_module()`.

### 3.3 Wire into the unified trainer

In `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`:

- Add `"hierarchical"` to the `model_type` choices (line ~204).
- Add an `elif args.model_type == "hierarchical":` branch that instantiates `RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint` with the new args:
  - `n_view_groups`
  - `n_view_layers`
  - `n_temporal_layers`
  - `n_joint_graph_layers`
  - `use_skeleton_graph`

### 3.4 Wire into evaluation

In `experiments/eval_full_metrics.py`:

- Import the hierarchical model class.
- Add `"hierarchical_pp"` to `MODEL_CLASSES` (line ~80-97).
- In `build_model`, route to the new class and set the extra hyper-parameters.

### 3.5 Add a dedicated smoke/full script

Create `experiments/train_hierarchical_pp_smoke_mpiinf3dhp.py` (smoke) and later `experiments/train_hierarchical_pp_full_mpiinf3dhp.py` (full). Pattern after `experiments/train_bayesian_tri_pp_full_mpiinf3dhp.py`, which calls the shared trainer with `--model_type hierarchical`.

### 3.6 Unit / smoke test

Create `tests/test_hierarchical_attention.py` to verify:

- Forward pass shape `(B, T, J, 3)` and `(B, V, J)` weights.
- Gradient flow through the three stages.
- Variable `n_view_groups` works for `V ∈ {4, 14}`.

---

## 4. Training & Evaluation Protocol

### 4.1 Datasets

Use the same canonical WebBridge NPZ layout (`TemporalClipDataset` in `motionflow_mv/data/temporal_clip_dataset.py`):

- **Train:**
  - `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz`
  - `data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz`
  - `data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz`
  - `data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz`
- **Val:**
  - `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz`
- **Smoke (fast):**
  - `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz` if available, or the full train set with `--train_samples 500 --epochs 5`.

### 4.2 Hyper-parameters

| Parameter | Anchor | Hierarchical proposed |
|-----------|--------|-----------------------|
| `d` | 64 | 64 |
| `residual_hidden` | 128 | 128 |
| `n_st_layers` | 2 | — (replaced by hierarchy) |
| `n_view_groups` | — | 2 or 4 |
| `n_view_layers` | — | 2 |
| `n_temporal_layers` | — | 2 |
| `n_joint_graph_layers` | — | 1 |
| `clip_len` | 13 | 13 |
| `batch_size` | 8 | 8 |
| `epochs` | 10 (robust re-train) | 10–20 |
| `lr` | 1e-3 | 1e-3 |
| `pp_loss_weight` | 0.1 | 0.1 |
| `cam_aug_pp` | 5.0 px | 5.0 px |
| `cam_aug_focal` | 0.01 | 0.01 |
| `cam_aug_schedule` | `intrinsics_curriculum` | `intrinsics_curriculum` |

### 4.3 Loss

Primary 3D MSE:

```python
criterion = nn.MSELoss()
loss = criterion(pred, yb)
```

Optional auxiliaries already supported by the trainer:
- PP correction loss (`args.pp_loss_weight > 0`)
- Reprojection loss (`motionflow_mv.losses.reprojection_loss`)
- Velocity loss (`motionflow_mv.losses.velocity_loss`)

For the first full run, keep the same auxiliaries as the anchor: PP loss only. Add velocity/reproj only if the smoke shows instability.

### 4.4 Metrics

Evaluate with `motionflow_mv.eval.metrics.compute_all_metrics` via `experiments/eval_full_metrics.py`:

- MPJPE (mm)
- PA-MPJPE (mm)
- PCK@50/100/150 mm
- PCK-AUC (0–150 mm)
- Bone-length error (if parents supplied)

### 4.5 Baseline comparison

Compare against the anchor checkpoint:

```bash
python experiments/eval_full_metrics.py \
    --model crossview_residual_pp \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --checkpoint outputs/ray_attention_temporal_crossview_residual_principal_point_robust_retrain.pth \
    --clip_len 13 --d 64 --residual_hidden 128 --n_st_layers 2 \
    --val_stride 50 \
    --output_json outputs/anchor_eval.json
```

And the new model:

```bash
python experiments/eval_full_metrics.py \
    --model hierarchical_pp \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --checkpoint outputs/ray_attention_hierarchical_pp_full_mpiinf3dhp.pth \
    --clip_len 13 --d 64 --residual_hidden 128 \
    --n_view_groups 2 --n_view_layers 2 --n_temporal_layers 2 --n_joint_graph_layers 1 \
    --val_stride 50 \
    --output_json outputs/hierarchical_pp_eval.json
```

---

## 5. Expected Gains and Risks

### 5.1 Expected gains

- **Cleaner view mixing:** grouping close cameras first reduces the impact of a single noisy view on the global triangulation. This should lower MPJPE on joints with high self-occlusion (wrists, ankles).
- **Temporal smoothing:** a dedicated temporal stage avoids the flat transformer “diluting” motion cues among 14×13 view-time tokens.
- **Skeleton regularization:** `GraphJointRelation` explicitly propagates anatomical constraints, which helps PA-MPJPE.
- **Modest parameter budget:** the hierarchy replaces, rather than stacks on top of, `st_transformer`. Expected model size stays under 300 k params, preserving the anchor’s speed advantage.
- **Target:** 0.3–0.8 mm improvement on clean MPI-INF-3DHP, i.e. **8.0–8.4 mm MPJPE**.

### 5.2 Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Contiguous view grouping is arbitrary. | Medium | Upgrade to geometry-aware grouping using camera positional encoding before the full run. |
| Graph layers add memory for large `V`. | Medium | Keep `n_joint_graph_layers=1` initially; use the existing sparse `edge_index` from `build_edge_index`. |
| Hierarchy is slower per forward than the flat transformer. | Low-Medium | Measure with `experiments/benchmark_runtime.py`; if >20% slower, consider reducing `d` to 32 for the smoke. |
| Overfitting on the small MPI-INF-3DHP train split. | Medium | Reuse the anchor’s augmentations and PP curriculum; run multi-seed (`experiments/run_repeated_seeds.py`). |
| Variable-view inference not yet supported. | Medium | Test with `VariableViewInferenceWrapper` from `motionflow_mv/fusion/variable_view_inference.py` only after the clean full run. |

---

## 6. Next Steps

1. **Implement the `FusionModule` wrapper and trainer wiring** (no GPU needed).
   - `motionflow_mv/fusion/ray_attention_hierarchical_view_temporal_joint_residual_principal_point_module.py`
   - update `motionflow_mv/fusion/__init__.py`
   - add `hierarchical` branch in `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
   - add `hierarchical_pp` in `experiments/eval_full_metrics.py`

2. **Run a CPU smoke test** to verify forward/backward and metric integration.

3. **Queue the smoke training** (`--epochs 5`, `--train_samples 500`) on the RTX 4090 only after the Bayesian Triangulation run finishes.

4. **If smoke MPJPE < 15 mm**, launch the full 10–20 epoch run with the same data/augmentation as the anchor.

5. **Compare full checkpoint** to the 8.75 mm anchor using `eval_full_metrics.py` and the robustness matrix script (`experiments/eval_robustness_matrix_pp_mpiinf3dhp.py`) for rotation/translation/focal/PP perturbations.

6. **If clean result ≤ 8.4 mm**, integrate the hierarchical backbone into the next high-priority directions: variable-view (`VariableViewInferenceWrapper`) and visibility-gated v2 (`motionflow_mv/models/crossview_residual_visibility_v2.py`).
