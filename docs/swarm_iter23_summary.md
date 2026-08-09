# Swarm Iteration 23 Summary

**Status:** Synthesis complete  
**Tracking issue:** #160 (v46 Sparse-View Generalization)  
**Branch:** `v46-svg`  
**Date:** 2026-08-09

---

## 1. Executive summary

Swarm iteration 23 focused on converging the `v18 → v22/v23 → v25` line of MotionFlow-MultiView models and locking down the data, benchmark, and test-infrastructure assumptions needed for the ICRA/CVPR 2027 paper. Ten agent reports landed in `docs/swarm_iter23/` covering: A800 training status, benchmark protocol, WebBridge mixed-data splits, Kinematic Anthropometric Prior (KAP) improvements, Neural Bundle Adjustment (NBA) fixes, related work, test gaps, v23 full-scale launch planning, v25 integration, and the open-issue roadmap.

The headline takeaway is that **v23 (v18 + KAP, no neural BA) is the immediate production candidate**, with v25 geometry fusion as the next integration target. Neural BA (v21) remains off the main path until the safety fixes in `docs/swarm_iter23/nba_fixes.md` are smoke-tested. v46 Sparse-View Generalization is queued as the next feature after the v23/v25 baseline stabilizes.

---

## 2. Reports synthesized

| Report | Owner scope | Key finding |
|---|---|---|
| `log_comparison.md` | Training status | v18 small first-epoch `val_MPJPE` = 20.24 mm; v23 small is training stably on GPU4/GPU6 (~1,850 steps, loss ≈ 7.2). No val yet. |
| `webbridge_data_report.md` | Data | Mixed WebBridge/H36M/MPI manifest has 271,365 samples over 45 files; H36M has 4 views, MPI has 14 views — view-count heterogeneity is real. |
| `benchmark_protocol.md` | Benchmark | Proposes a single YAML-driven protocol with H36M S1→S9→S11 and MPI S2→official test server; frame-weighted aggregation; no test-set tuning. |
| `v23_fullscale_plan.md` | Launch | v23 full-scale on A800: d=128, batch=16, 10k samples/epoch, 60 epochs, ~24–30 GB memory, 30–60 h wall time. |
| `v25_integration_plan.md` | Integration | v25 geometry fusion inserts after v18/KAP and before the ST transformer; start from a v23 checkpoint, freeze v18 for one epoch. |
| `kap_improvements.md` | KAP v22 | Recommends adaptive per-sample bone prior, weighted angle loss, and bone-length preservation on the residual. |
| `nba_fixes.md` | NBA v21 | Three fixes: structure-first BA ordering, compact axis-angle rotation descriptor, detach + residual-improvement gate. |
| `related_work_survey.md` | Literature | Recent CVPR/ICRA/RA-L work supports variable-view fusion, ray-based lifting, and skeleton priors; strongest links to MV-SSM, RUMPL, DeProPose. |
| `test_gaps.md` | Testing | 76 of 137 fusion modules have no direct tests; critical gaps in robust triangulation, differentiable BA, camera helpers, and principal-point ray variants. |
| `roadmap_comment.md` | Roadmap | 21 open issues, 32 open PRs, A800 fully loaded; v23 small is the gate for full-scale launch; v21 is blocked until smoke tests pass. |

---

## 3. Synthesis

### 3.1 Baseline decision: v23 is the fork point

- **v18** is the validated deformable-cross-view-attention baseline (20.24 mm first-epoch val).
- **v23 = v18 + KAP, no neural BA**. Early loss curves are stable and deterministic across replicas. KAP adds a small absolute-loss offset (~1.5) but no instability.
- **v21 (neural BA) regressed to 128.27 mm** and is parked. The fixes in `nba_fixes.md` must be implemented and smoke-tested before any re-introduction.
- **v25** is the next integration target *after* v23 is validated; it adds geometry attention + learned depth triangulation + analytic GeoBA, intended to supersede v21.

### 3.2 Data and benchmark readiness

The WebBridge mixed split is now characterized:
- Train: 181,104 samples / Val: 90,261 samples.
- View heterogeneity: H36M clips are 4-view, MPI clips are 14-view.
- The recommended protocol avoids test-set leakage and uses frame-weighted averages; final MPI numbers require the official evaluation server.

### 3.3 Technical risks

1. **KAP may not lift much.** `kap_improvements.md` identifies three concrete upgrades if v23 small shows limited gain or instability.
2. **NBA can still regress.** The v21 regression is mitigated by design in v25 (analytic GeoBA), but the v21 fixes are also available if we ever re-enable the learned camera head.
3. **Test coverage is thin.** 76 untested fusion modules create regression risk for v25/v46. `test_gaps.md` prioritizes robust triangulation, differentiable BA, and camera helpers.
4. **GPU bottleneck.** A800 is fully booked; v46 smoke must wait for 4090 or a free A800 GPU.

### 3.4 Connection to v46 Sparse-View Generalization

v46 (#160) builds on the same variable-view training already present in v23/v25. The WebBridge mixed split (4-view H36M + 14-view MPI) is the natural training ground for sparse-view dropout. The benchmark protocol supports `MPJPE@k` reporting, which v46 will use to demonstrate generalization at 2–3 views.

---

## 4. Recommendations and next steps

1. **Gate v23 full scale on first-epoch val.** Launch `scripts/launch_v23_a800_fullscale.sh` as soon as a GPU frees and the v23 small `val_MPJPE` is ≤ v18 small baseline.
2. **Hold v21.** Do not re-enable neural BA until the `nba_fixes.md` changes are smoke-tested and show no regression.
3. **Prepare v25 integration.** Wire `MultiViewGeometryFusionV25` into `OmniMultiViewFusionV5` per `v25_integration_plan.md`; start from the v23 checkpoint and freeze v18 for one epoch.
4. **Close test gaps incrementally.** Prioritize robust triangulation, differentiable BA, and camera helper tests before landing v25/v46.
5. **Queue v46 smoke only after v23/v25 baseline stabilizes.** The v46 view-dropout module is lightweight, but it should not compete with baseline validation for GPU time.

---

## 5. Open questions

- Does v23 small beat v18 small at first-epoch val? Await GPU4/GPU6 results.
- Will the KAP improvements in `kap_improvements.md` be needed, or is the current KAP sufficient?
- Can the v25 geometry-fusion block be integrated without increasing per-step latency beyond the A800 budget?
- Which untested fusion modules are exercised by v25 and therefore must be tested first?

---

*Generated by Agent-20 as a synthesis of the ten reports in `docs/swarm_iter23/` for the v46 Sparse-View Generalization swarm (#160).*
