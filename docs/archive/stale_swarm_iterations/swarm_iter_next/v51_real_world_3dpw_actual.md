# v51 Real-World 3DPW Actual Geometric Self-Refinement

**Focus area:** real_world_3dpw_actual  
**Built on:** v46 sparse-view generalization + v48 domain generalization + v50 self-evolution feedback head  
**Tracking issue:** (v51, to be opened after v50 smoke results)

## 1. Problem

v48 domain generalization reduces the studio-to-3DPW actual gap with FiLM/GRL/DDWL, but the remaining error on 3DPW actual is still dominated by two real-world failure modes: (1) mild camera-calibration and lens distortion artifacts that violate the ideal pinhole assumptions used in triangulation, and (2) over-confident uncertainty on the out-of-domain sequences. v50 SEFH gives us per-view reliability and per-joint log-variance, but it is trained once and not re-optimized for each specific real-world clip at test time. v51 closes that last mile by adding a **test-time geometric self-refinement block targeted at 3DPW actual**.

## 2. Module: `RealWorld3DPWActualRefinerV51`

### 2.1 Architecture

A lightweight differentiable refinement head that runs only for sequences flagged as 3DPW actual (domain id `D=3`).

**Inputs** (per frame):
- Current 3-D pose estimate `P ∈ R^(J×3)`
- 2-D keypoints `x_v ∈ R^(J×2)` for each available view `v`
- Camera matrices `K_v, R_v, t_v`
- v50 SEFH per-view reliability `r_v ∈ [0,1]` and per-joint log-variance `σ_j`
- v48 domain embedding `d ∈ R^h`

**Refinement network**: a 2-layer MLP (`hidden=64`) maps the concatenation `[d; mean_v(r_v · ρ_v); var_v(ρ_v)]` to a pose residual `ΔP` and an uncertainty recalibration scale `α ∈ R^J`, where `ρ_v` is the normalized per-view reprojection residual. A zero-initialized residual gate `g(·)` keeps the module identity at startup.

**Test-time refinement loop** (fixed 3 steps at inference):
1. Compute weighted reprojection loss `L_repr = Σ_v r_v · ||Π_v(P) - x_v||^2`.
2. Add a skeleton prior `L_bone` from v40 and a frame-to-frame smoothness term `L_temp` over a 5-frame window.
3. Update `P ← P - η ∇_P (L_repr + λ_bone L_bone + λ_temp L_temp)`.
4. Return refined pose `P'` and recalibrated uncertainty `σ' = α · σ`.

During training the same loop is unrolled once so gradients reach the MLP and the v50 reliability weights.

### 2.2 New config flags and defaults

| Flag | Type | Default |
|---|---|---|
| `use_v51_real_world_3dpw_actual` | bool | `False` |
| `v51_rw3dpw_hidden` | int | `64` |
| `v51_rw3dpw_num_layers` | int | `2` |
| `v51_rw3dpw_test_time_steps` | int | `3` |
| `v51_rw3dpw_reproj_weight` | float | `1.0` |
| `v51_rw3dpw_bone_weight` | float | `0.5` |
| `v51_rw3dpw_temporal_weight` | float | `0.3` |
| `v51_rw3dpw_refinement_lr` | float | `0.01` |
| `v51_rw3dpw_uncertainty_temperature` | float | `1.0` |
| `loss.v51_rw3dpw_loss_weight` | float | `0.01` |

### 2.3 Loss term

```
L_rw3dpw = loss.v51_rw3dpw_loss_weight · [
    L_reproj(P', x; r) / J
  + λ_bone · L_bone(P')
  + λ_temp · L_temp(P')
  + γ · |mean_j(σ'_j) - mean_j(||P'_j - P_gt,j||)|
]
```

The last term is a calibration loss that keeps the recalibrated uncertainty honest on labeled 3DPW actual frames.

### 2.4 Evaluation metric

- `MPJPE@k` for `k = 2, 3, 4, full` on 3DPW actual mode.
- Studio-val `MPJPE@full` to ensure no regression.
- 3DPW actual-to-studio full-view gap (target: reduce by ≥2 mm).
- Spearman(`σ'`, joint error) on 3DPW actual (target: > 0.35).

### 2.5 Expected MPJPE impact

- 3DPW actual `MPJPE@2`: **−4 to −7 mm**
- 3DPW actual `MPJPE@full`: **−2 to −3 mm**
- Studio validation `MPJPE@full`: **±0.5 mm**

### 2.6 Main risk

**Risk:** Test-time refinement may overfit to the training camera distribution or collapse to the identity mapping, leaving 3DPW actual performance unchanged.  
**Mitigation:** zero-initialized residual gate, clamp refinement magnitude to `≤ 5 cm`, cap test-time steps at 3, and freeze the v50 base for the first epoch while only the v51 head trains.

## 3. Why this fits v51

v50 turns reprojection residuals into reliability; v48 turns domain labels into domain-invariant features. v51 uses both to run a final, real-world-specific geometric self-correction, making the pipeline genuinely self-evolving on the hardest deployment domain. It is optional and identity-at-init, so it can be smoke-tested on the local RTX 4090 without disturbing the v46/v48 baseline.
