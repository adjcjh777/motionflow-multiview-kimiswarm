# Roadmap Comment — swarm_iter23

_Generated: 2026-08-08 03:35 UTC._

This is a snapshot of all open GitHub issues/PRs plus the current A800-D training queue, intended to be posted as a roadmap update in the tracking issue.

## 1. Open Issues

Total: **21**

| # | Title | Updated | Labels |
|---|-------|---------|--------|
| 89 | [v22] Kinematic Anthropometric Prior (KAP) implementation and validation | 2026-08-08 | none |
| 88 | [iter-21] v18 full-scale + v17/v18 small monitoring | 2026-08-08 | none |
| 87 | [Iter-21] Integrate v17-v22 prototypes and monitor full-scale v11 | 2026-08-08 | none |
| 85 | v7/v8 iteration tracking: full-precision DLT, domain embedding, curriculum variable views, PA loss | 2026-08-07 | none |
| 84 | v6: geometry-aware view-mask fix + Perceiver aggregator + monotonic multi-view loss | 2026-08-07 | none |
| 83 | v5 set-transformer / camera-conditioned variable-view fusion | 2026-08-07 | enhancement |
| 75 | [Tracking] Swarm iteration 19 – A800 scaling, multi-dataset, and v3 design | 2026-08-07 | none |
| 73 | [Tracking] OmniMultiViewFusion v2 full d=128 training run | 2026-08-07 | tracking, research, swarm-iter18, omniview |
| 41 | [webbridge_dataset_integration] Mixed 17-joint trainer with cross-dataset consistency loss | 2026-08-06 | none |
| 25 | iter16 next-candidate tracking | 2026-08-07 | none |
| 21 | 20-agent swarm synthesis and next-iteration roadmap | 2026-08-06 | none |
| 20 | [Plan] 20-agent swarm next-iteration review and top-5 priorities | 2026-08-05 | enhancement, planning |
| 19 | Iter10-next: focal-aware intrinsic correction | 2026-08-05 | none |
| 16 | Phase 2 (Iter10): Temporal ray-attention residual fusion for multi-view human pose | 2026-08-04 | none |
| 15 | Phase 1: define HumanMotionIR and single-view zero-regression passthrough | 2026-08-04 | none |
| 12 | Iter9: use 3D GT / pseudo-label or skeleton prior to beat DLT | 2026-08-04 | none |
| 10 | Iter8: establish clear advantage over DLT (3D GT / pseudo-label / motion prior) | 2026-08-04 | none |
| 8 | Iter7: beat DLT with temporal consistency or 3D GT supervision | 2026-08-04 | none |
| 6 | Iter6: beat DLT on real Shelf data (temporal / 3D residual / better regularization) | 2026-08-04 | none |
| 4 | Iteration 4：提升 AttentionFusion 在真实数据上追平/超越 DLT | 2026-08-04 | none |
| 3 | Iteration 3：跨视角行人匹配 + DLT/Attention 对比 | 2026-08-04 | none |

## 2. Open Pull Requests

Total: **32**

| # | Title | Updated | Labels |
|---|-------|---------|--------|
| 69 | feat(iter17): mixed-dataset balanced sampling | 2026-08-07 | none |
| 68 | [iter-17] SplatV2 view-dependent covariance smoke test | 2026-08-07 | none |
| 67 | [iter17] Confidence-aware view dropout smoke prototype | 2026-08-07 | none |
| 66 | feat(iter17): cross-view graph attention CPU smoke test | 2026-08-07 | none |
| 65 | [iter-17] visibility-uncertainty-v1 CPU smoke test | 2026-08-07 | none |
| 64 | feat(iter17): attention-entropy-regularization CPU smoke | 2026-08-07 | none |
| 63 | iter17: temporal velocity + acceleration loss smoke | 2026-08-07 | none |
| 62 | feat(iter17): epipolar-bias-v2-lite prototype | 2026-08-07 | none |
| 61 | [iter-17] Extended camera perturbation curriculum smoke | 2026-08-07 | none |
| 60 | iter17: EMA checkpoint save/load smoke prototype | 2026-08-07 | none |
| 59 | [iter-17] graph-joint-relation CPU smoke prototype | 2026-08-07 | none |
| 58 | Add iter17 camera-conditioned-pp CPU smoke prototype | 2026-08-07 | none |
| 57 | [iter-17] Cross-view contrastive SSL smoke test | 2026-08-07 | none |
| 56 | [iter17] physics-motion-prior CPU smoke test | 2026-08-07 | none |
| 55 | feat(iter17): deeper ST-attention smoke prototype | 2026-08-07 | none |
| 54 | [iter17] Kinematic-chain-constraints smoke prototype | 2026-08-07 | none |
| 53 | feat(iter17): CPU-only smoke test for Bayesian triangulation v3 | 2026-08-07 | none |
| 52 | [Iter-17] Semi-supervised pseudo-labeling smoke prototype | 2026-08-07 | none |
| 51 | [iter17] Adaptive-scale cross-view spatial pyramid smoke prototype | 2026-08-07 | none |
| 47 | feat(temporal): deeper residual-gated temporal attention | 2026-08-07 | none |
| 46 | feat: lightweight late-layer epipolar geometry bias (epipolar_bias_v2_lite_pp) | 2026-08-07 | none |
| 45 | [Design] Shelf/Campus domain adaptation | 2026-08-07 | none |
| 44 | feat(crossview): visibility + uncertainty v1 full run | 2026-08-07 | none |
| 43 | realtime-kd-student-iter16: Knowledge-distilled lightweight student for real-time compression | 2026-08-07 | none |
| 42 | Bayesian triangulation ablation flags and paper-story design doc | 2026-08-07 | none |
| 40 | Attention-entropy regularisation for interpretable multi-view fusion | 2026-08-07 | none |
| 37 | SplatV2: view-dependent Gaussian covariance for multi-view pose fusion | 2026-08-07 | none |
| 36 | Adaptive scale-gated cross-view spatial pyramid fusion | 2026-08-07 | none |
| 35 | Generalize 6-axis robustness matrix to all registered models | 2026-08-07 | none |
| 34 | Semi-supervised pseudo-labeling trainer for MPI-INF-3DHP | 2026-08-07 | none |
| 33 | feat: kinematic-chain constraints auxiliary loss | 2026-08-07 | none |
| 32 | SSL pretraining with cross-view contrastive loss (#23 #25) | 2026-08-07 | none |

## 3. A800-D Training Queue (current snapshot)

- **v23** = v18 + KAP, no neural BA — running on **GPU4 & GPU6**, waiting for first-epoch `val_MPJPE`.
- **v18 full** — running on **GPU5**.
- **v11 full** — running on **GPU7**.
- **v21** = neural BA — regressed to **128.27 mm** and was stopped.
- **v24** = v18 + fixed BA + KAP — prepared but **not launched**; all A800 GPUs are busy.

## 4. Roadmap & Next Steps

### Immediate (next 24–48 h)
1. **Watch v23 small first-epoch `val_MPJPE` on GPU4/GPU6.** If it meets or beats the v18 small baseline (~20 mm), queue the v23 full-scale run on the first free GPU (see `docs/swarm_iter23/v23_fullscale_plan.md`).
2. **Let v18 (GPU5) and v11 (GPU7) full-scale runs finish** to preserve baseline checkpoints.
3. **Do not restart v21** until the neural-BA safety fixes in `docs/swarm_iter23/nba_fixes.md` are implemented and smoke-tested.
4. **Keep v24 on hold** until v23 is validated and the fixed neural-BA branch is proven stable in a small run.

### Short-term (this week)
- Land/review the most promising iter17 prototype PRs once their smoke tests are complete (e.g. balanced mixed-dataset sampling #69, view-dependent covariance #68, visibility+uncertainty #44, epipolar-bias v2 lite #62).
- Use issues #88/#87/#89 as the v18→v22→v23 convergence tracker; close or merge duplicates and update their checklists.
- Apply the KAP improvements in `docs/swarm_iter23/kap_improvements.md` if v23 small shows KAP instability or limited gain.

### Medium-term
- Re-introduce neural BA only after the fixes in `docs/swarm_iter23/nba_fixes.md` show no regression in small-scale runs.
- Archive/close stale iteration issues that no longer reflect the current v18-centered line (notably older issues #4, #3, #6, #8, #10, #12).

## 5. Single Next Concrete Step

**Queue `scripts/launch_v23_a800_fullscale.sh` on the first free A800 GPU as soon as the GPU4/GPU6 v23 small run reports a first-epoch `val_MPJPE` ≤ v18 small baseline, and do not launch v24 until that v23 result is in.**
