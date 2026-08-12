# Design Report: Full MPI-INF-3DHP Skeleton Graph Joint Relation (v2)

## 1. Objective

Implement a skeleton-graph-aware variant of the CamPE residual temporal model that is hard-coded for the full 28-joint MPI-INF-3DHP skeleton.  The existing `RayAttentionFusionModelTemporalResidualCamPEGraph` is generic and defaults to the H36M 17-joint skeleton; this v2 model makes the MPI skeleton a first-class citizen and adds lightweight, reusable skeleton-regularisation helpers.

## 2. What changed

### 2.1 New model file

`motionflow_mv/fusion/ray_attention_temporal_residual_campe_graph_mpi_v2_model.py`

Contains:

- `GraphJointRelationMPIV2` — improved edge-conditioned message-passing block:
  - Vectorised edge projection (single `nn.Linear` instead of looping over edge types).
  - Edge-type embedding added to the attention gate so bone/symmetry/cross-view edges can gate differently.
  - LayerNorm + residual per message-passing step.
- `RayAttentionFusionModelTemporalResidualCamPEGraphMPIV2` — full model that:
  - Defaults to `j=28` and the MPI-INF-3DHP 28-joint skeleton.
  - Validates that the provided skeleton matches the joint count.
  - Replaces the dense `joint_attn` with the MPI graph module.
  - Exposes `bone_length_loss(pred, target)` and `symmetry_loss(pred)` helpers for skeleton-aware training losses.

### 2.2 Key differences from the generic CamPE graph model

| Aspect | `ray_attention_temporal_residual_campe_graph_model.py` | v2 |
|---|---|---|
| Default skeleton | H36M 17-joint | MPI-INF-3DHP 28-joint |
| Edge projection | Loop over 3 edge-type MLPs | Single shared linear + edge-type embedding |
| Skeleton validation | None | Asserts `len(parents) == j` |
| Skeleton losses | Not included | `bone_length_loss`, `symmetry_loss` |
| Target dataset | H36M / generic | MPI-INF-3DHP |

## 3. Validation

Run the built-in CPU smoke test:

```bash
source .venv/bin/activate
KMP_DUPLICATE_LIB_OK=TRUE python -m motionflow_mv.fusion.ray_attention_temporal_residual_campe_graph_mpi_v2_model
```

Expected output:

```
CamPE + MPI Graph v2 model sanity check passed
bone_length_loss=..., symmetry_loss=...
```

The smoke test checks:
- Forward pass with `B=2, T=5, V=4, J=28` produces shapes `(B, T, J, 3)` and `(B, T, V, J)`.
- Gradients flow through the whole model.
- `bone_length_loss` and `symmetry_loss` run without errors and return non-negative scalars.

## 4. Expected impact

- **Occlusion robustness:** occluded MPI joints can borrow evidence from anatomical neighbours via bone/symmetry edges.
- **Skeleton plausibility:** adding `bone_length_loss` and `symmetry_loss` during training should reduce anatomically implausible poses.
- **Clean MPI baseline:** provides a ready-to-train MPI-INF-3DHP model with the full 28-joint skeleton, avoiding the empty-skeleton fallback used previously.
- **Minimal blast radius:** the new file does not modify `ray_attention_temporal_residual_model.py`, `ray_attention_temporal_residual_campe_model.py`, or `graph_joint_relation.py`.

## 5. Next steps / blockers

- Full training would require an MPI-INF-3DHP training script that wires `bone_length_loss` and `symmetry_loss` into the objective (analogous to `experiments/train_ray_attention_temporal_residual_campe_graph_mpiinf3dhp_fullskeleton.py`).
- The MPI-INF-3DHP 28-joint parent/symmetry lists in `graph_joint_relation.py` are the best available topology; validation against the official MPI-INF-3DHP joint ordering is recommended before final benchmarking.
- No GPU or long training was run; only CPU smoke-test validation was performed.
