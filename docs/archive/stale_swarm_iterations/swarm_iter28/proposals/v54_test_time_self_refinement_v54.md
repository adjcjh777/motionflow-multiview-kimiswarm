# v54: Test-Time Self-Refinement with Physical Feedback (TTSR-v54)

**Task identifier:** `design_v54_test_time_self_refinement`  
**Status:** Proposal (no code yet)  
**Builds on:** v25 geometry fusion, v45 adaptive geometry fusion, v46 sparse-view generalization, v47 temporal aggregation, v48 domain generalization, v49-Lite, v50 Self-Evolution Feedback Head, v51 Cross-Domain Sparse-View Reliability, v52 Uncertainty-Weighted Triangulation, v53 Physical-Space Calibration.

## 1. Motivation

The pipeline now ends with v53 Physical-Space Calibration, a single feed-forward block that uses v52 uncertainty weights, floor-plane and bone-length priors. The paper narrative is an *optimized* motionflow pipeline: multi-view video → human pose extraction → multi-view fusion and calibration → physical-space alignment → optimized motionflow. That last stage needs a **closed-loop refinement** step that re-evaluates the calibrated pose against both multi-view evidence and physical invariants.

**v54** adds a lightweight, warm-startable **test-time self-refinement** module that treats the v53 output as an initial guess and performs `K` learned refinement steps. It reuses v52 per-view precision weights and v53 physical feedback (floor height, bone-scale factors) so corrections remain physically consistent. The correction and step-gate heads are zero-initialized, making the module a strict no-op at initialization; it can be dropped onto any pretrained v53 checkpoint without regression.

## 2. Architecture and equations

Let the pose at step `k` be `P^{(k)} ∈ R^{B×T×J×3}` with `P^{(0)}` the v53-calibrated pose.

**Per-view reprojection residual.**

```text
π_v(P^{(k)}_{t,j}) = Project(K_{t,v}, R_{t,v}, t_{t,v}, P^{(k)}_{t,j})
r^{(k)}_{t,v,j} = || π_v(P^{(k)}_{t,j}) - x^{2d}_{t,v,j} ||_2
```

**Weighted per-joint residual** using v52 weights `w` and `view_mask`:

```text
e^{(k)}_{t,j} = Σ_v w_{t,v,j} * view_mask_{t,v} * r^{(k)}_{t,v,j} / (Σ_v w_{t,v,j} + ε)
```

**Joint feature token.** For each joint `j`:

```text
f^{(k)}_j = Linear( [ P^{(k)}_{t,j},
                       e^{(k)}_{t,j},
                       height_{t,j} - h_t,              # v53 floor hint
                       ||b_{t,j}|| - s_{t,bone(j)},       # v53 bone-scale hint
                       P^{(k)}_{t+1,j} - 2P^{(k)}_{t,j} + P^{(k)}_{t-1,j},  # acceleration
                       ∇_P e^{(k)}_{t,j} ] ) ∈ R^{d}
```

A two-layer skeleton graph network (`v54_ttsr_hop_distance = 1`) produces per-joint hidden states `H`. A zero-initialized head outputs a per-joint direction `ΔP` and step gate `g`:

```text
ΔP^{(k)}, g^{(k)} = ZeroInitHead(GNN(f^{(k)}))
P^{(k+1)} = P^{(k)} + α * sigmoid(g^{(k)}) * ΔP^{(k)}
```

`α = v54_ttsr_step_size` (default `0.1`). At init, `ΔP = 0` and `sigmoid(g) ≈ 0`, so `P^{(1)} = P^{(0)}`. At inference we unroll `v54_ttsr_num_steps = 3` steps; at training we use a single step to keep the graph small.

## 3. Inputs/outputs

**Forward signature**

```python
pred_3d_ref, ttsr_loss = ttsr(
    pred_3d_init,        # (B, T, J, 3)    v53-calibrated pose
    points_2d,           # (B, T, V, J, 2)
    K, R, t,             # (B, T, V, 3, 3/3)
    view_mask,           # (B, T, V)
    v52_weights,         # (B, T, V, J)
    v52_log_precision,   # (B, T, V, J)
    v53_floor_height,    # (B, T)
    v53_bone_scale,      # (B, T, n_bones)
    domain_id,           # (B,)
)
```

**Outputs**

* `pred_3d_ref`: `(B, T, J, 3)` — refined 3-D pose.
* `ttsr_loss`: scalar — `λ_reproj L_reproj + λ_bone L_bone + λ_floor L_floor + λ_temp L_temporal`, evaluated on the refined pose.

## 4. Config flags

```python
use_v54_test_time_self_refinement: bool = False
v54_ttsr_hidden: int = 64
v54_ttsr_n_layers: int = 2
v54_ttsr_num_heads: int = 4
v54_ttsr_num_steps: int = 3
v54_ttsr_train_steps: int = 1
v54_ttsr_hop_distance: int = 1
v54_ttsr_step_size: float = 0.1
v54_ttsr_dropout: float = 0.1
v54_ttsr_use_v52_weights: bool = True
v54_ttsr_use_v53_floor: bool = True
v54_ttsr_use_v53_bone_scale: bool = True
v54_ttsr_identity_init: bool = True
v54_ttsr_residual_gate_init: float = -6.0
v54_ttsr_loss_weight: float = 0.01
v54_ttsr_reproj_weight: float = 1.0
v54_ttsr_bone_weight: float = 0.1
v54_ttsr_floor_weight: float = 0.01
v54_ttsr_temporal_weight: float = 0.05
v54_ttsr_warmup_epochs: int = 1
```

## 5. Integration point

Instantiate the module in `OmniMultiViewFusionV5.__init__` and call it **after** the v53 PSC block. Pass the v52 weights/log-precision and v53 floor/bone-scale tensors forward from the preceding blocks. Add `v54_loss` to the auxiliary loss dictionary with weight `v54_ttsr_loss_weight`.

## 6. Expected MPJPE impact

* **Smoke (RTX 4090, 50–100 samples):** no regression with `identity_init=True` and `loss_weight=0` (within `1e-3` mm).
* **Medium (500–2k samples):** 1–2 mm improvement over v53 PSC on WebBridge/H36M, mainly on frames with residual reprojection or bone-length errors.
* **Full mixed-domain:** 2–4 mm improvement over the strongest v50/v51/v52/v53 stack, with larger gains on sparse-view `MPJPE@2/3`.
* **3DPW actual / cross-domain:** 3–5 mm improvement from adapting to domain-shifted camera/keypoint noise without 3-D labels.

## 7. Risks

See `docs/swarm_iter28/reports/agent_test_time_self_refinement_v54_risks.md` for detailed risks and mitigations.

## 8. 5-step implementation plan

1. **Geometry helper:** ensure `motionflow_mv/utils/geometry.py` has a batched `project_points` for `(B, T, V, J, 3)` points and `(B, T, V, 3, 3/4)` cameras.
2. **Module stub:** implement `TestTimeSelfRefinementV54` in `motionflow_mv/fusion/test_time_self_refinement_v54.py` with the skeleton-graph network, zero-initialized correction/gate heads, and self-supervised loss.
3. **Model wiring:** add the v54 toggle block to `OmniMultiViewFusionV5.__init__` and insert the forward call immediately after the v53 PSC block; accumulate `v54_loss` into the auxiliary loss dictionary.
4. **Smoke test:** create `configs/benchmark_v54_ttsr_smoke.yaml` and `scripts/run_v54_ttsr_smoke_local_4090.sh`; verify identity-at-init and that `loss_weight=0` does not change val_MPJPE.
5. **Ablation + A800 queue:** add an entry to `scripts/launch_v33_a800_queue.py`; compare `num_steps={1,3,5}`, `step_size={0.05,0.1,0.2}`, and physical-feedback ablations against the v53 PSC baseline; report per-domain and `MPJPE@k` metrics.
