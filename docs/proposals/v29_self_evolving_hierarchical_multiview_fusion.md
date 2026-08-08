# v29: Self-Evolving Hierarchical Multi-View Fusion (SEH-MV)

**Task identifier:** `design_v29_self_evolving_hierarchical_multiview_fusion`  
**Depends on:** v25 (`docs/proposals/v25_multiview_geometry_fusion.md`), v26 (`docs/proposals/v26_temporal_geometry_fusion.md`), v27 (`docs/proposals/v27_test_time_self_evolution.md`), v28 (`docs/proposals/v28_physical_space_alignment.md`)  
**Status:** Design / Candidate direction  

---

## 1. Problem

v25 introduced learned depth triangulation and geometry bundle adjustment, but the current pipeline still has three weaknesses that limit CVPR/ICRA-quality results:

1. **Single-scale view aggregation.** v25 fuses all views at a fixed resolution. Fine-grained joint details and coarse body topology are mixed in the same token stream, hurting both accuracy and generalisation.
2. **Fixed inference pass.** The model predicts once and stops. When 2D detections are noisy or views are few (2–4), the triangulated pose can be improved by iterative self-consistency without any extra data.
3. **Weak physical-space constraints.** v28 aligns poses to a learned floor plane and bone-length prior, but it is applied as a post-hoc residual. The geometry-fusion stage itself does not exploit floor/contact or temporal consistency cues.

## 2. Proposed method

SEH-MV keeps the v25 ray-token + epipolar cross-view attention backbone and adds three lightweight, flag-gated components:

### 2.1 Hierarchical multi-scale view encoder (`HierarchicalViewEncoderV29`)

Split the per-view 2D feature pyramid into three scales:

- **Joint scale** (J tokens): per-joint 2D keypoint embeddings.
- **Part scale** (P ≈ 6 tokens): group joints into torso/head/limbs, aggregate with a small graph attention.
- **Body scale** (1 token): global body pose embedding.

Each scale has its own cross-view attention block. The outputs are fused with learned scale weights before the depth-triangulation head. This is analogous to a multi-scale spatial pyramid, but applied to the *view dimension* rather than the image dimension.

**Why it helps:** Few-view joints benefit from coarse part-level redundancy; many-view joints benefit from fine joint-level detail. Hierarchical aggregation reduces over-smoothing.

### 2.2 Test-time self-evolution loop (`TestTimeSelfEvolutionV29`)

Building on v27, add a *training-free* loop at inference:

```text
For k = 1..K (K=3 by default):
    1. Reproject current 3D pose to every view.
    2. Compute per-view/joint residual r_vj.
    3. Update confidence w_vj = softmax(-|r_vj| / sigma).
    4. Re-run v25 depth-triangulation head with updated confidences.
    5. Optional: apply v28 physical-space alignment to keep floor/bone constraints.
```

The loop is wrapped in a new flag `--use_test_time_self_evolution_v29` and is parameterised by:

- `v29_tte_iters` (default 3)
- `v29_tte_sigma_reproj` (default 5 px)
- `v29_tte_use_physical_space_alignment` (default True)

**Why it helps:** It converts the model from a single-shot estimator to a self-correcting estimator, similar in spirit to the self-evolution ideas in Qwen3-style iterative reasoning but applied to geometric consistency.

### 2.3 Physical-space aware temporal regulariser (`PhysicalSpaceTemporalLossV29`)

During training, add a lightweight loss that penalises:

- **Foot-floor penetration:** `max(0, -y_foot)` after ground-plane alignment.
- **Unphysical bone-length change across time:** `|L(t) - L(t+1)|^2` where `L(t)` is the vector of bone lengths at frame t.
- **Center-of-mass jitter:** `|com(t) - com(t+1)|^2`.

The loss is gated by `--v29_physical_loss_weight` (default 0.01).

**Why it helps:** It gives the geometry-fusion stage an explicit physical prior, reducing the overfitting we observe in v25 after epoch 1.

## 3. Architecture file plan

```text
motionflow_mv/fusion/self_evolving_hierarchical_multiview_v29.py
    HierarchicalViewEncoderV29
    TestTimeSelfEvolutionV29
    PhysicalSpaceTemporalLossV29

experiments/train_omniview_fusion_v5_webbridge_multi.py
    add flags:
        --use_hierarchical_multiview_v29
        --v29_part_scale_groups
        --v29_tte_iters
        --v29_tte_sigma_reproj
        --v29_tte_use_physical_space_alignment
        --v29_physical_loss_weight

motionflow_mv/losses/physical_space_temporal_v29.py
    PhysicalSpaceTemporalLossV29
```

## 4. Experimental plan

### 4.1 Data

Use the existing WebBridge mixed loader (`configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`) which already provides multi-view H36M + MPI-INF-3DHP. No new dataset download is required.

### 4.2 Smoke / local 4090

- 1 epoch, 100 samples, batch size 4 to verify forward/backward and flag wiring.
- Target: train loss decreases, no OOM, log contains `v29_hierarchical_encoder` and `v29_tte` shapes.

### 4.3 Small-scale local 4090

- 20 epochs, 2000 samples, batch size 16.
- Compare against v25 baseline (current run in `outputs/omniview_fusion_v25_geometry_fusion_small_local_4090.log`).
- Success criterion: best val_MPJPE < v25 best val_MPJPE (target < 30 mm locally).

### 4.4 Full-scale A800

Once an A800 GPU frees:
- 20 epochs, full WebBridge mixed train set, batch size 32–64.
- Compare against A800 v25 small (current best 18.31 mm on `v25_geometry_fusion_small_gpu7`).
- Success criterion: best val_MPJPE < 18.31 mm or overfitting is delayed by at least 3 epochs.

### 4.5 Ablation matrix

| Variant | Hierarchical | TTE | Physical loss | Expected role |
|--------|--------------|-----|---------------|---------------|
| v29a   | Yes          | No  | No            | isolate hierarchical encoder gain |
| v29b   | Yes          | Yes | No            | add self-evolution at inference |
| v29c   | Yes          | Yes | Yes           | full SEH-MV |
| v29d   | No           | Yes | Yes           | isolate physical loss + TTE |

## 5. Success criteria

- [ ] Local smoke passes.
- [ ] Local small-scale best val_MPJPE < v25 baseline (target < 30 mm).
- [ ] A800 full-scale best val_MPJPE ≤ 18.31 mm or overfitting delayed ≥ 3 epochs.
- [ ] A clean PR is merged to `main` behind feature flags.

## 6. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Hierarchical encoder adds memory. | Use smaller part scale and gradient checkpointing; smoke first. |
| TTE loop slows inference. | Make K configurable; default K=3 adds ~10 % latency. |
| Physical loss destabilises training. | Start with weight 0.001 and warm up. |
| WebBridge subset mismatch. | Use the same manifest as v25. |

## 7. Next immediate steps

1. Implement `HierarchicalViewEncoderV29` and wire the flag.
2. Add a smoke test `scripts/run_v29_smoke_4090.sh`.
3. Run smoke on local 4090.
4. If smoke passes, start a 20-epoch small-scale local run.
5. Open issue + PR to track.
