# v27: Multi-Scale Spatial Refinement

**Task identifier:** `design_v27_multiscale_spatial_refinement`  
**Depends on:** v25 (`docs/proposals/v25_multiview_geometry_fusion.md`), v26 (`docs/proposals/v26_temporal_geometry_fusion.md`), v22 KAP (`docs/proposals/v22_kinematic_anthropometric_prior.md` if present)  
**Status:** Design proposal — no code implemented yet.

## 1. Problem

v25/v26 fuse multi-view geometry at the **joint level**: each joint is triangulated and refined using rays, epipolar constraints, and (in v26) a small temporal window. This is strong for per-joint reprojection but weak when a joint is locally under-constrained:

- **Local ambiguity in few views.** With only 2–4 views, a single joint’s ray intersection can be noisy, and the per-joint head has no explicit way to borrow information from neighbouring joints (e.g. wrist ↔ elbow).
- **No explicit spatial-scale hierarchy.** v22 KAP enforces global skeleton structure, but it operates later in the pipeline and only via bone-length/angle priors. There is no light-weight, learned refinement that reasons at **part scale** (arm, leg, torso) and **whole-body scale** before the final residual head.
- **Residual MLP is joint-wise.** The deterministic residual MLP in `OmniMultiViewFusionV5` processes each joint independently after feature pooling; it does not explicitly encode multi-scale spatial dependencies.

v27 adds a small, warm-startable spatial-refinement block that operates on 3D poses at multiple spatial scales to regularise locally ambiguous joints while preserving fine detail.

## 2. Proposed method

### 2.1 Module: `MultiscaleSpatialRefinementV27`

**New file:** `motionflow_mv/fusion/multiscale_spatial_refinement_v27.py`

```text
MultiscaleSpatialRefinementV27(
    j: int = 17,                       # number of joints
    d_part: int = 64,                  # part-token dimension
    d_global: int = 128,               # whole-body token dimension
    part_groups: List[List[int]],      # e.g. [[0,1,2], [3,4,5,6], ...]
    n_part_layers: int = 2,            # transformer layers over part tokens
    n_global_layers: int = 1,         # transformer layers over whole-body token + parts
    dropout: float = 0.1,
    max_residual_m: float = 0.10,      # clamp per-joint residual update
)
```

### 2.2 Inputs / outputs

```python
pred_3d_ref, spatial_loss = multiscale_spatial_refinement(
    pred_3d,        # (B, T, J, 3)  3D pose from v26 (or v25)
    confidence,     # (B, T, J)    optional per-joint confidence / visibility
)
```

- `pred_3d_ref`: `(B, T, J, 3)` — spatially refined 3D pose.
- `spatial_loss`: scalar — optional smoothness term on part-scale offsets (can be `0.0` by default).

### 2.3 Multi-scale architecture

The block pools joints into a fixed set of **body-part tokens**, refines them, pools further into a **whole-body token**, then redistributes corrections back to each joint.

1. **Joint → part pooling**
   - For each predefined part group `p`, average the joint positions (and optional confidence-weighted mean) to form a part center.
   - Embed the part center with a small MLP: `part_token_p = MLP([mean(X_p); radius(X_p)])`, where `radius` is the mean distance of joints in the part from the part center. This captures both location and scale.

2. **Part-scale refinement**
   - Run `n_part_layers` self-attention layers over the `P` part tokens.
   - Output: refined part offsets `Δ_part_p` (initialised to zero).

3. **Whole-body scale**
   - Pool all part tokens to a single global token via attention or simple mean.
   - Run `n_global_layers` over `[global_token, part_tokens]` so each part receives global context (e.g. standing vs. sitting pose).

4. **Part → joint redistribution**
   - Each joint receives the corrected offset of its containing part.
   - A per-joint residual MLP blends the part-level correction with the original joint position.
   - The final output is clamped: `pred_3d_ref = pred_3d + clamp(Δ, -max_residual_m, +max_residual_m)`.

### 2.4 Where it fits in `omniview_fusion_v5.py`

Insert **after** the v26 temporal geometry fusion block and **before** the v20 residual/diffusion refiner and v22 KAP. The v26 hook currently ends around the block cited in `docs/proposals/v26_temporal_geometry_fusion.md`, §4.3:

```python
# motionflow_mv/fusion/omniview_fusion_v5.py
# after v26 geometry fusion
if self.use_multiscale_spatial_refinement_v27:
    pred_3d_gn, spatial_loss_v27 = self.multiscale_spatial_refinement_v27(
        pred_3d_gn.view(B, T, J, 3),
        confidence=...,
    )
    pred_3d_gn = pred_3d_gn.view(B * T, J, 3)
```

The block is a pure residual, so with zero-initialised output projection it starts as an no-op and a v26 checkpoint can be warm-started with `use_multiscale_spatial_refinement_v27=True`.

## 3. Expected impact

| Metric | Estimate vs. v26 baseline | Rationale |
|--------|----------------------------|-----------|
| `val_MPJPE` | −5 % to −10 % | Part/global regularisation reduces local joint drift, especially on noisy/occluded joints. |
| 2-view MPJPE | −8 % to −12 % | Largest gain: few-view triangulation is under-constrained and benefits most from spatial context. |
| 4-view MPJPE | −5 % to −8 % | Moderate gain; geometry is already fairly strong. |
| 8-view / 14-view | −2 % to −5 % | Diminishing returns; many-view geometry already suppresses drift. |

These estimates assume v26 reaches a comparable `val_MPJPE` to v18 (≈20 mm). If v26 itself under-performs, the gains from v27 may be smaller because spatial refinement cannot fully compensate for broken geometry.

## 4. Implementation cost

| Item | Estimate |
|------|----------|
| New module | ~200–250 lines in `motionflow_mv/fusion/multiscale_spatial_refinement_v27.py` |
| Plumbing in `omniview_fusion_v5.py` | ~20 lines (toggle, instantiation, forward hook) |
| Tests | ~80–120 lines in `tests/test_multiscale_spatial_refinement_v27.py` |
| Training time increase | < 5 % per step (tiny MLP/attention over J=17/28 joints) |
| Data needs | None beyond current WebBridge / H36M / MPI pipelines |
| Parameters | < 100 k additional params with `d_part=64`, `d_global=128` |

## 5. Risks / mitigation

| Risk | Detection | Mitigation |
|------|-----------|------------|
| Over-smooths fine joint detail (e.g., wrist rotation). | Compare per-joint error maps; watch large joints (elbow/knee) improve but small joints (wrist/ankle) regress. | Keep `max_residual_m=0.05–0.10 m`; make part groups fine enough to preserve local detail. |
| Redundant with v22 KAP. | Toggle `use_kinematic_anthropometric_prior_v22` on/off with v27. | Apply v27 **before** KAP; let KAP only regularise remaining skeleton violations. |
| Part groups mismatch dataset skeletons. | Run smoke test on both J=17 (H36M) and J=28 (MPI) skeletons. | Part groups are a constructor argument; provide per-dataset defaults. |
| v26 baseline is unstable; v27 cannot fix geometry errors. | If v26 `val_MPJPE` > 21 mm, stop v27 and debug v26 first. | Gate v27 behind decision-matrix gates G1 and G4 from `docs/proposals/v27_next_iteration_decision_matrix.md`. |
| Extra loss term swamps MPJPE. | Monitor `spatial_loss` magnitude; should be < 1 % of total loss at init. | Default `v27_spatial_loss_weight=0.0`; enable only after first smoke. |

## 6. Minimal experiment plan

### 6.1 New flags / config names

Add to the model constructor and YAML configs (e.g. `configs/train_multiscale_spatial_refinement_smoke.yaml`):

```yaml
use_multiscale_spatial_refinement_v27: true
v27_part_groups: [[0,1,2], [3,4,5,6], [7,8,9,10], [11,12,13], [14,15,16]]
v27_d_part: 64
v27_d_global: 128
v27_n_part_layers: 2
v27_n_global_layers: 1
v27_max_residual_m: 0.10
v27_spatial_loss_weight: 0.0   # enable after smoke
v27_use_global_scale: true
```

### 6.2 Smoke test

1. **Unit test**
   ```bash
   pytest tests/test_multiscale_spatial_refinement_v27.py -q
   ```
   Covers: forward shape `(B,T,J,3) → (B,T,J,3)`, zero-residual at init, gradient flow, invalid part-group index errors, J=17 and J=28.

2. **Training smoke (RTX 4090)**
   ```bash
   python experiments/train.py \
     --config configs/train_multiscale_spatial_refinement_smoke.yaml \
     --data.h36m.small True \
     --n_views 4 \
     --max_steps 50 \
     --use_multiscale_spatial_refinement_v27 True \
     --v27_spatial_loss_weight 0.0
   ```
   Success: loss is finite, `val_MPJPE` is not NaN, and the module runs faster than 5 % overhead vs. the same run with `use_multiscale_spatial_refinement_v27=False`.

3. **First real run**
   - Warm-start from the best v26 small checkpoint.
   - Train on H36M + WebBridge small subset with 4 views.
   - Compare `val_MPJPE` at the end of epoch 1.
   - If gain > 3 %, launch full 2/4/8/14 view variable-view evaluation.

## 7. Simpler variant (if full block is too heavy)

If the two-scale (part + global) transformer is unstable, fall back to a **single-scale part-level residual** (`PartSpatialRefinerV27`): only steps 1, 2, and 4 above, with no global token. This removes the global attention layer, cuts parameters by ~40 %, and still gives most of the spatial regularisation benefit.

## 8. References

- `motionflow_mv/fusion/omniview_fusion_v5.py`
- `motionflow_mv/fusion/multiview_geometry_fusion_v25.py`
- `motionflow_mv/fusion/temporal_geometry_fusion_v26.py`
- `motionflow_mv/fusion/adaptive_hierarchical_multiscale_fusion.py` (existing feature-level multi-scale fusion, not directly reused but conceptually related)
- `docs/proposals/v25_multiview_geometry_fusion.md`
- `docs/proposals/v26_temporal_geometry_fusion.md`
- `docs/proposals/v27_next_iteration_decision_matrix.md`
