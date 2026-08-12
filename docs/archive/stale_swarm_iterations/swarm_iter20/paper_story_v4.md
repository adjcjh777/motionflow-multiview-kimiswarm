# MotionFlow-MultiView v4 — Paper Story and Novelty Positioning

> Tracking issue: #76  
> Branch: `feat/swarm-iter20-v4`  
> Last updated: 2026-08-07

---

## 1. One-sentence thesis

**We present a view-mask-aware adaptive multi-view fusion system that makes calibrated multi-view 3D human pose estimation robust to missing, noisy, or miscalibrated views by treating geometry as the predictor and learning only the structured residual error: a visibility-gated cross-view graph attention, an adaptive view selector, a calibration-correction head, and a skeleton-graph residual refiner—wrapped in a single warm-startable module that closes the 2–3 view inference gap on the path to ICRA/CVPR 2027.**

---

## 2. Narrative: from v2/v3 to v4

### 2.1 What we already showed (v2/v3)

MotionFlow-MultiView demonstrated that *triangulate first, then learn the residual* is the right decomposition for compact, accurate, multi-view human pose estimation:

- A **Bayesian precision-weighted triangulation** step uses predicted 2D covariances to weight a differentiable DLT.
- An **adaptive Gauss-Newton refinement** keeps the solution geometrically consistent.
- A **principal-point / intrinsic correction head** makes the system robust to realistic calibration drift.
- A small **residual MLP** corrects structured detector bias.

The current anchor single model reaches **9.03 mm** MPJPE on MPI-INF-3DHP S2/Seq1; an ensemble reaches **8.35 mm**.

### 2.2 What remains unsolved (the v4 target)

Real-world capture breaks the clean 4-view rig assumption in three ways:

1. **Variable views.** The best v2/v3 model catastrophically fails when fewer than four views are available: 2-view MPJPE ≈ 1990 mm, 3-view  1620 mm on H36M v2. The system must be reliable at k=2,3,4.
2. **Occlusion and view dropout.** Joint occlusion and view dropout are the largest remaining degradations (30 % view dropout → 18.15 mm; 30 % joint occlusion → 16.99 mm).
3. **Calibration drift.** Rotation (0.5° → 16.89 mm) and focal-length (1 % → 19.13 mm) errors are still headline weaknesses.

### 2.3 The v4 claim

v4 addresses these three gaps with a **unified, modular, view-mask-aware fusion architecture**:

- **View-mask-aware visibility gating** replaces a single per-view visibility head with a per-joint context visibility head that reasons across views and predicts explicit uncertainty.
- **Adaptive view selection** learns a budgeted view subset per joint via Gumbel-softmax during training and hard top-k during inference.
- **Rotation correction head** predicts a bounded SO(3) residual per view before triangulation, directly targeting rotation-robustness.
- **Skeleton-graph residual refiner** replaces the dense residual MLP with a graph neural network over bone/symmetry/self-loop edges, enforcing anatomical consistency.
- **Kinematic-chain graph refiner** provides a final optional skeleton-aware temporal pass on the output 3D pose.
- **Attention-entropy regularization** sharpens the per-view weight distribution and improves interpretability.
- **Calibration perturbation curriculum** progressively increases rotation, focal, and principal-point noise during training.

All modules are individually togglable and warm-startable from v2/v3 checkpoints (`strict=False`), so v4 is a **direct, low-risk evolution**, not a ground-up redesign.

---

## 3. Module-to-paper-section mapping

| v4 module | Paper section | Role in the story |
|-----------|---------------|-------------------|
| **View-mask-aware visibility gating v2** | 4.1 “Visibility-aware view masking” | Learns per-joint occlusion masks across views; closes joint-occlusion and view-dropout gap. |
| **Adaptive view selector** | 4.2 “Adaptive budgeted view selection” | Selects a small reliable subset of views per joint; enables 2–3 view inference. |
| **Rotation correction head** | 4.3 “Calibration-robust triangulation” | Predicts a bounded SO(3) residual before triangulation; fixes rotation drift. |
| **Principal-point + focal correction** | 4.3 (continued) | Existing v2/v3 self-calibration; extended with focal correction enabled by default. |
| **Skeleton-graph residual refiner** | 4.4 “Anatomical residual refinement” | Replaces dense residual MLP with bone/symmetry graph propagation. |
| **Kinematic-chain graph refiner** | 4.5 “Optional kinematic post-refinement” | Final temporal skeleton pass on 3D pose; targets distal-limb errors. |
| **Attention-entropy regularization** | 4.6 “Regularization losses” | Sharpens per-view weight distribution; auxiliary loss. |
| **Calibration perturbation curriculum** | 5.1 “Training protocol” | Progressive rotation/focal/PP augmentation; robustness training. |
| **Variable-view inference wrapper** | 5.2 “Evaluation protocol” | Standardises MPJPE@k for k=2..V. |

### Suggested paper outline

1. **Introduction.** Multi-view pose is essential for robotics; DLT is brittle; end-to-end learning discards geometry; our view-mask-aware adaptive fusion combines the best of both.
2. **Related work.** Triangulation, learnable triangulation, visibility/occlusion handling, calibration-aware pose, graph neural networks for skeletons, robot motion pipelines.
3. **Preliminaries.** Differentiable weighted DLT and HumanMotionIR plugin interface (reused from v2/v3).
4. **Method.**
   - 4.1 Visibility-aware view masking
   - 4.2 Adaptive budgeted view selection
   - 4.3 Calibration-robust triangulation (PP/focal/rotation correction)
   - 4.4 Anatomical residual refinement
   - 4.5 Optional kinematic post-refinement
   - 4.6 Regularization losses (entropy, reprojection, epipolar, bone-length)
   - 4.7 Training and inference protocols
5. **Experiments.**
   - 5.1 Datasets and metrics
   - 5.2 MPI-INF-3DHP accuracy and ablations
   - 5.3 Human3.6M / WebBridge cross-dataset validation
   - 5.4 Robustness matrix (noise, occlusion, view dropout, calibration)
   - 5.5 Variable-view MPJPE@k curve
   - 5.6 Runtime and system integration
6. **Discussion.** What works, what does not, future work.

---

## 4. Quantitative claims we can defend once v4 trains complete

These are the strongest claims v4 is designed to support. Each is grounded in an existing v2/v3 number and a v4 target from `docs/v4_architecture_design_proposal.md`.

1. **MPI-INF-3DHP single-model accuracy under 8.6 mm.**  
   *Anchor:* v2/v3 single model 9.03 mm; ensemble 8.35 mm.  
   *v4 target:* single-model MPJPE < 8.6 mm on MPI-INF-3DHP S2/Seq1.  
   *Why defensible:* v4 keeps the same geometry-first decomposition and only adds lightweight, warm-startable modules.

2. **Variable-view inference no longer catastrophically fails at k=2,3.**  
   *Anchor:* H36M v2: 2-view ~1990 mm, 3-view ~1620 mm.  
   *v4 target:* MPJPE@2 < 50 mm and MPJPE@3 < 30 mm on variable-view evaluation.  
   *Why defensible:* Adaptive view selection + view-mask-aware visibility + confidence fallback to triangulation are explicitly designed for missing views.

3. **Occlusion and view-dropout robustness improve.**  
   *Anchor:* 30 % view dropout → 18.15 mm; 30 % joint occlusion → 16.99 mm.  
   *v4 target:* 30 % view dropout < 16.3 mm; 30 % joint occlusion < 16.0 mm.  
   *Why defensible:* Per-joint context visibility with uncertainty scaling directly models occluded views.

4. **Calibration robustness improves for rotation and focal length.**  
   *Anchor:* rot_0.5° → 16.89 mm; focal_1 % → 19.13 mm.  
   *v4 target:* rot_0.5° < 14 mm; focal_1 % < 15 mm.  
   *Why defensible:* Rotation correction head + perturbation curriculum directly target these failure modes.

5. **Compact and warm-startable.**  
   *Anchor:* v2/v3 models are < 1.1 M parameters.  
   *v4 target:* Full v4 with all toggles enabled remains < 2.0 M parameters and loads v2/v3 checkpoints with `strict=False`.  
   *Why defensible:* All new modules are small MLP/graph heads; no new large backbone is introduced.

---

## 5. Ablation table outline

The following table will be filled after the v4 ablation matrix (Section 6.3 of `docs/v4_architecture_design_proposal.md`) completes on A800.

| Run | Visibility v2 | Skeleton residual | KC refiner | Adaptive views | Rotation corr. | Entropy | MPI clean (mm) | rot_0.5° (mm) | view_drop_30 (mm) | MPJPE@2 (mm) |
|-----|:-------------:|:-----------------:|:----------:|:--------------:|:--------------:|:-------:|---------------:|--------------:|------------------:|-------------:|
| G (v3 baseline) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 9.03 | 16.89 | 18.15 | ~1990 |
| A (v4 full) | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | TBD | TBD | TBD | TBD |
| B | ✗ | ✓ | ✓ |  | ✓ | ✗ | TBD | TBD | TBD | TBD |
| C | ✓ | ✗ | ✓ | ✗ | ✓ |  | TBD | TBD | TBD | TBD |
| D | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | TBD | TBD | TBD | TBD |
| E | ✓ | ✓ |  | ✓ | ✓ | ✗ | TBD | TBD | TBD | TBD |
| F | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | TBD | TBD | TBD | TBD |

*Values are placeholders until v4 training and evaluation complete.*

---

## 6. Risks and mitigations

| Risk | Status | Mitigation |
|------|--------|------------|
| New modules destabilise training | Open | All modules optional; warm-start from 9.03 mm checkpoint; freeze encoder for first 5 epochs. |
| Variable-view k=2 still fails | Open | Confidence fallback to triangulation when active views < `min_views`; adaptive view selection hard top-k. |
| Rotation correction overfits | Open | Bound with `tanh`; curriculum increases perturbation only after 10 epochs. |
| Skeleton residual over-smooths | Open | Keep a small dense residual branch; graph refiner only adds correction. |

---

## 7. Related files

- Architecture design: `docs/v4_architecture_design_proposal.md`
- System story: `docs/paper_story_system_v2.md`
- Conference story: `docs/icra_cvpr_2027_paper_story.md`
- v4 model: `motionflow_mv/fusion/omniview_fusion_v4.py`
- v4 trainer: `experiments/train_omniview_fusion_v4_webbridge_multi.py`
- v4 eval: `experiments/eval_omniview_fusion_v4_mpiinf3dhp.py`
- v4 tests: `tests/test_omniview_fusion_v4.py`
