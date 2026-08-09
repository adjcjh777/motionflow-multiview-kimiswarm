# v53: Test-Time Self-Refinement (TTSR-v53)

**Task identifier:** `design_v53_test_time_self_refinement`  
**Status:** Proposal (no code yet)  
**Builds on:** v25 geometry fusion, v45 adaptive geometry fusion, v46 sparse-view generalization, v47 temporal aggregation, v48 domain generalization, v49-Lite, v50 Self-Evolution Feedback Head, v51 Cross-Domain Sparse-View Reliability, v52 Uncertainty-Weighted Triangulation.

## 1. Motivation

The current MotionFlow-MultiView pipeline produces a 3D pose through a single feed-forward pass: triangulation → Gauss–Newton → v45/v46/v47/v48/v49/v50/v51 refinement → v52 uncertainty-weighted triangulation → residual MLP. The paper narrative is an *optimized* motionflow pipeline: multi-view video → human pose extraction → multi-view fusion/calibration → physical-space alignment → optimized motionflow. This implies a final closed-loop refinement stage that re-evaluates the pose against the multi-view evidence after all learned fusion has run.

**v53** adds a lightweight, warm-startable **test-time self-refinement** module that treats the output of v52 as an initial guess and performs a small number of learned gradient-free refinement steps. The module re-uses the per-view, per-joint uncertainty weights learned by v52, computes reprojection residuals, and propagates corrections along the skeleton via a graph network. Because the correction and step-gate heads are zero-initialized, the module is a strict no-op at initialization, so it can be dropped onto any pretrained v52 checkpoint without regression.

## 2. Module overview

**File:** `motionflow_mv/fusion/test_time_self_refinement_v53.py`

```text
TestTimeSelfRefinementV53(
    d=64,
    n_views=4,
    hidden=64,
    n_layers=2,
    num_steps=3,
    use_v52_weights=True,
    identity_init=True,
)
```

### 2.1 Inputs and outputs

**Forward signature**

```python
pred_3d_ref, ttsr_loss = ttsr(
    pred_3d_init,    # (B, T, J, 3)   initial 3-D pose (output of v52 / residual MLP)
    points_2d,       # (B, T, V, J, 2) calibrated 2-D keypoints
    K,               # (B, T, V, 3, 3) intrinsics
    R,               # (B, T, V, 3, 3) rotations
    t,               # (B, T, V, 3)   translations
    view_mask,       # (B, T, V)       bool / float mask (True/1 = visible)
    v52_weights,     # (B, T, V, J)    optional per-view per-joint weights from v52
    v52_log_precision, # (B, T, V, J)  optional log-precision from v52
)
```

**Outputs**

* `pred_3d_ref`: `(B, T, J, 3)` — refined 3-D pose.
* `ttsr_loss`: scalar — auxiliary self-supervised energy (reprojection + bone-length + temporal terms).

### 2.2 Architecture and equations

Let the current pose estimate at iteration `k` be `P^{(k)} ∈ R^{B×T×J×3}` with `P^{(0)}` the input `pred_3d_init`.

**Per-view reprojection residual.** For each view `v` and joint `j`:

```text
π_v(P^{(k)}_{t,j}) = Project( K_{t,v}, R_{t,v}, t_{t,v}, P^{(k)}_{t,j} )
r^{(k)}_{t,v,j} = || π_v(P^{(k)}_{t,j}) - x^{2d}_{t,v,j} ||_2
```

`r^{(k)} ∈ R^{B×T×V×J}`.

**Weighted per-joint residual.** If `v52_weights` is provided, we re-weight residuals; otherwise uniform over visible views:

```text
w_{t,v,j} = v52_weights_{t,v,j} * view_mask_{t,v}
e^{(k)}_{t,j} = Σ_v w_{t,v,j} * r^{(k)}_{t,v,j} / (Σ_v w_{t,v,j} + ε)
```

`e^{(k)} ∈ R^{B×T×J}`.

**Joint feature token.** For each joint `j` we concatenate:

```text
f^{(k)}_j = Linear( [ P^{(k)}_{t,j},
                       e^{(k)}_{t,j},
                       ∇_P e^{(k)}_{t,j},        # finite-difference or autograd hint (3-D)
                       b_{t,j},                    # bone direction from parent
                       l_{t,j},                    # bone-length residual
                       a_{t,j} ] ) ∈ R^{d}
```

where `b_{t,j} = P^{(k)}_{t,j} - P^{(k)}_{t,parent(j)}`, `l_{t,j} = ||b_{t,j}|| - μ_bone(j)`, and `a_{t,j} = P^{(k)}_{t+1,j} - 2P^{(k)}_{t,j} + P^{(k)}_{t-1,j}` (zero-padded) is the temporal acceleration.

**Skeleton graph network.** Two graph-convolution / transformer layers over the kinematic skeleton restrict message passing to joints within a fixed graph distance (`v53_ttsr_hop_distance = 2`):

```text
H^{(l+1)} = GNN_l( H^{(l)}, A )   with A_{ij} = 1 if skeleton_distance(i,j) ≤ 2
```

**Correction head (identity at init).** Two zero-initialized linear layers output a per-joint direction `ΔP ∈ R^{B×T×J×3}` and a per-joint step gate `g ∈ R^{B×T×J,1}`:

```text
ΔP^{(k)}, g^{(k)} = ZeroInitHead( GNN(f^{(k)}) )
P^{(k+1)} = P^{(k)} + α * sigmoid(g^{(k)})  ΔP^{(k)}
```

The scalar `α` is fixed to `v53_ttsr_step_size` (default `0.1`). Because `ΔP` and `g` are zero at init, `P^{(1)} = P^{(0)}`, i.e. the module is identity-at-init and warm-startable from any v52 checkpoint.

**Iterative test-time refinement.** At inference, the same network is unrolled `v53_ttsr_num_steps` times (default `3`) with shared weights. At training, only a single forward step is used to keep the computational graph small; the loss is computed on `P^{(1)}`.

### 2.3 Self-supervised auxiliary loss

During training the module is supervised by the downstream 3-D MPJPE, plus three optional self-supervised terms:

```text
L_ttsr = λ_3d * MPJPE(P*, P_gt)
       + λ_reproj * L_reproj(P*; x_2d, K, R, t, w)
       + λ_bone * L_bone(P*)
       + λ_temp * L_temporal(P*)
```

* `L_reproj` is the weighted reprojection MSE using the v52 weights.
* `L_bone` is a skeleton bone-length prior (zero-centered at the dataset mean).
* `L_temporal` is an finite-difference acceleration penalty.

All self-supervised terms use the current refined pose, not the initial guess, so the network learns to actually reduce the energy it is optimizing.

## 3. Integration into `OmniMultiViewFusionV5`

### 3.1 New toggles

```python
use_v53_test_time_self_refinement: bool = False,
v53_ttsr_hidden: int = 64,
v53_ttsr_n_layers: int = 2,
v53_ttsr_num_heads: int = 4,
v53_ttsr_num_steps: int = 3,
v53_ttsr_hop_distance: int = 2,
v53_ttsr_step_size: float = 0.1,
v53_ttsr_use_v52_weights: bool = True,
v53_ttsr_use_v52_log_precision: bool = True,
v53_ttsr_identity_init: bool = True,
v53_ttsr_loss_weight: float = 0.01,
v53_ttsr_reproj_weight: float = 1.0,
v53_ttsr_bone_weight: float = 0.1,
v53_ttsr_temporal_weight: float = 0.05,
```

### 3.2 Wiring

The module is instantiated in `OmniMultiViewFusionV5.__init__` when `use_v53_test_time_self_refinement` is true and called **after** the residual MLP (and after any v52 UWT output is reshaped to `(B, T, J, 3)`):

```python
if self.use_v53_test_time_self_refinement:
    pred_3d, v53_loss = self.test_time_self_refinement_v53(
        pred_3d.view(B, T, J, 3),
        points_2d=points_2d.view(B, T, V, J, 2),
        K=K_corrected.view(B, T, V, 3, 3),
        R=R.view(B, T, V, 3, 3),
        t=t.view(B, T, V, 3),
        view_mask=view_mask_flat.view(B, T, V),
        v52_weights=v52_weights,          # from v52 UWT
        v52_log_precision=v52_log_precision,
    )
    pred_3d = pred_3d.view(B * T, J, 3)
```

The auxiliary loss `v53_loss` is added to the total training loss with weight `v53_ttsr_loss_weight`.

## 4. Expected MPJPE impact

* **Smoke (RTX 4090, 50–100 samples):** no regression when `v53_ttsr_identity_init=True` and `v53_ttsr_loss_weight=0`; identity verification should hold within `1e-3` mm.
* **Medium (500–2k samples):** 1–2 mm improvement over v52 UWT alone on WebBridge/H36M, mainly on frames with large reprojection residuals after the residual MLP.
* **Full mixed-domain (10k+ samples):** 2–4 mm improvement over the strongest v50/v51/v52 stack, with larger gains on sparse-view settings (`MPJPE@2` and `MPJPE@3`) where the iterative correction redistributes error along the skeleton.
* **3DPW actual / cross-domain:** 3–5 mm improvement because the self-supervised reprojection and bone-length terms adapt to domain-shifted camera/keypoint noise without requiring 3-D labels.

## 5. Risks

See `docs/swarm_iter27/reports/agent_test_time_self_refinement_v53_risks.md` for detailed risks and mitigations. Key concerns are inference latency from iterative unrolling, gradient instability through the reprojection Jacobian, double-counting with v27/v29 test-time self-evolution, and overfitting to the residual-MLP output manifold.

## 6. 5-step implementation plan

1. **Geometry helper:** add a small batched `project_points` utility to `motionflow_mv/utils/geometry.py` if it does not already exist, supporting `(B, T, V, J, 3)` points and `(B, T, V, 3, 3/4)` camera parameters.
2. **Module stub:** implement `TestTimeSelfRefinementV53` in `motionflow_mv/fusion/test_time_self_refinement_v53.py` with the skeleton-graph network, zero-initialized correction/gate heads, and the self-supervised loss.
3. **Model wiring:** add the v53 toggle block to `OmniMultiViewFusionV5.__init__`, and insert the forward call after the residual MLP (where `pred_3d` is already `(B*T, J, 3)`). Accumulate `v53_loss` into the auxiliary loss dictionary.
4. **Smoke test:** create `configs/benchmark_v53_ttsr_smoke.yaml` and `scripts/run_v53_ttsr_smoke_local_4090.sh`; verify identity-at-init and that enabling v53 with `loss_weight=0` does not change val_MPJPE.
5. **Ablation + A800 queue:** add an entry to `scripts/launch_v33_a800_queue.py`; compare `num_steps={1,3,5}`, `use_v52_weights={True,False}`, and `step_size={0.05,0.1,0.2}` against the v52 UWT baseline; report per-domain and `MPJPE@k` metrics.
