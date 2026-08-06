<!--
Summary (produced 2026-08-04 by literature-survey subagent):
This file surveys CVPR/ICRA 2025-2026 multiview human pose/motion work, identifies
ten remaining gaps, and maps each gap to a concrete motionflow-multiview
opportunity. Main takeaway: geometry-aware learned fusion is now the consensus
direction, but most methods still lack (i) calibrated cross-dataset
generalization, (ii) lightweight plug-in deployment for robotics, and
(iii) explicit per-joint uncertainty. Our ray-attention v3 design already
covers the core fusion gap; the biggest differentiators to add are
temporal/bone-length priors, uncertainty-aware HumanMotionIR, and rigorous
cross-calibration ablation.
-->

# Multiview Human Pose / Motion Literature Gap Analysis
## CVPR / ICRA 2025-2026

**Scope:** Calibrated and weakly-calibrated multi-view 2D-to-3D human pose / motion estimation, with emphasis on methods that can feed a modular robotics/perception pipeline.

**Survey sources:** arXiv API (2025-2026 pre-prints), DBLP/CVPR 2025 open-access listings, and the project's prior swarm reports (`docs/swarm_iter3/`, `docs/swarm_iter4/`).

---

## 1. Recent literature overview (2025-2026)

| Paper / Venue | What it does | Relevance to us |
|---|---|---|
| **MV-SSM** (Chharia et al., CVPR 2025) | Multi-View State Space Model for 3D pose; replaces attention with Mamba-like scan over views. | Confirms the field is moving beyond vanilla transformers to *structured* view fusion; still regresses 3D joints. |
| **MVDoppler-Pose** (Choi et al., CVPR 2025) | mmWave + RGB multi-modal multi-view pose for occluded walking. | Robotics-relevant, but domain-specific sensor; not a general calibrated-RGB solution. |
| **DisPOSE** (arXiv 2026) | Self-supervised diffusion for multi-view 3D pose. | Reduces 3D-GT need, but diffusion cost and synthetic-to-real domain gap are open. |
| **RUMPL** (arXiv 2025) | Ray-based transformer for universal 2D-to-3D pose lifting. | Very close in spirit to our `ray_attention` line; emphasizes ray embeddings. |
| **COMPOSE** (arXiv 2026) | Hypergraph cover optimization over multi-view 2D detections. | Strong geometric prior, but optimization-based and slower; not a learned plugin. |
| **SkelSplat** (arXiv 2025) | Differentiable Gaussian rendering for multi-view pose. | Uses rendering loss; heavy compared to our DLT-based fusion. |
| **From Sparse to Dense / DenseWarper** (arXiv 2026) | Spatio-temporal fusion with dense warping. | Highlights temporal + dense correspondence as a frontier. |
| **Unconstrained Multi-view Human Pose with Algebraic Priors** (arXiv 2026) | Calibration-free pose from algebraic constraints. | Addresses uncalibrated settings, but assumes specific geometric priors. |
| **DeProPose / RapidPoseTriangulation** (arXiv 2025) | Adaptive multi-view fusion and fast whole-body triangulation. | Show learned fusion and fast geometry are both active sub-problems. |
| **HeatFormer** (arXiv 2024) | Multi-view human mesh recovery with neural optimizer. | Represents the parametric-body track; complements joint-level fusion. |

*Note on ICRA 2025:* ICRA 2025 contained relatively few pure multi-view *human pose* papers; most vision-for-robotics work in this area appears as RSS/HUMANOIDS workshops or as applied perception papers. The robotics gap is therefore not a lack of pose methods, but a lack of pose methods designed for downstream robot consumption (uncertainty, metric scale, real-time plugin operation).

---

## 2. Key trends

1. **Geometry-aware learned fusion is now the consensus.** Flattening projection matrices is out; ray/camera embeddings plus differentiable triangulation are in (MV-SSM, RUMPL, our `ray_attention_v3`).
2. **State-space and diffusion are emerging alternatives.** MV-SSM trades attention for linear-complexity scans; DisPOSE uses diffusion for self-supervision. Both trade simplicity for scalability or label efficiency.
3. **Temporal and dense correspondence are gaining attention.** DenseWarper-style spatio-temporal warping suggests per-frame triangulation is leaving motion information on the table.
4. **Parametric body recovery is a parallel track.** HeatFormer-style SMPL fitting operates on the same multi-view evidence but produces a body mesh rather than skeleton points.
5. **Self-supervised / weakly-supervised methods are rising.** DisPOSE and algebraic-prior methods reduce dependence on 3D GT, but evaluation on real robotic data is sparse.

---

## 3. Identified gaps

### Gap 1 — Cross-calibration generalization
Most 2025-2026 methods train and test on the same camera rig (Human3.6M, Shelf, Campus). Real robots change cameras, lenses, and baselines. Few papers report cross-dataset MPJPE after training on a *different* rig.

### Gap 2 — Calibrated plug-in operation
Methods are usually monolithic pipelines. There is little work on a *lightweight, interchangeable fusion plugin* that fits into an `HumanMotionIR`-style pipeline with explicit input/output scale contracts.

### Gap 3 — Explicit outlier/occlusion handling on real data
Synthetic dropout and Gaussian noise are common, but controlled real-data experiments with sparse 2D outliers (e.g., detector swaps, lens flare, motion blur) are rare.

### Gap 4 — Temporal / motion consistency
The majority of papers are per-frame. Methods that do use temporal information rarely combine it with geometry-aware fusion in a simple residual form.

### Gap 5 — Parametric body recovery after joint fusion
Joint-level triangulation and SMPL fitting are treated as separate stages. There is a gap for a method that uses the *same per-view weights* from learned fusion to drive a lightweight multi-view SMPL fit.

### Gap 6 — Per-joint, per-view uncertainty for downstream robotics
Few multi-view pose methods export actionable uncertainty (variance, per-view weight, reprojection residual) that a robot controller can threshold or fuse with other sensors.

### Gap 7 — Real-time / embedded deployment
Diffusion, Gaussian rendering, and large transformers are accurate but expensive. Robotics needs small networks (MBs, not GBs) running at camera frame rate.

### Gap 8 — Synthetic-to-real transfer without finetuning
Several papers generate synthetic training data, but the jump to real cameras usually requires real finetuning. Domain-agnostic camera embeddings that remove this need are underexplored.

### Gap 9 — Multi-person multi-view association
Single-person fusion is well studied; associating and triangulating multiple interacting people across views remains brittle, especially in sports/crowded robotics settings.

### Gap 10 — Weak- or no-3D-GT training on robot-relevant motions
Large 3D datasets are dominated by everyday standing/walking. Dynamic robot teleoperation motions are under-represented, limiting direct supervised training.

---

## 4. What this means for motionflow-multiview

Our current best model, `motionflow_mv/fusion/ray_attention_v3_model.py`, already targets the core geometry-aware fusion trend:

- ✅ **Gap 1 (cross-calibration)** → camera-conditioned ray embeddings; next step is explicit cross-dataset ablation (train Shelf, test Campus/H36M).
- ✅ **Gap 2 (plug-in)** → `FusionModule` plugin interface with `requires_calibration`, `input_scale`, `output_scale`.
- ✅ **Gap 3 (outliers)** → weighted-DLT attention head; already evaluated with synthetic/real outliers.
- ️ **Gap 4 (temporal)** → `temporal_refiner` exists but is not integrated into `ray_attention_v3`; add a light temporal/bone-loss auxiliary head.
- ⚠️ **Gap 5 (parametric body)** → GVHMR/ScoreHMR adapters exist, but multi-view SMPL fitting after fusion is not yet implemented.
- ⚠️ **Gap 6 (uncertainty)** → `HumanMotionIR` can be extended with `per_view_weights`, `reprojection_residual`, and `joint_std` fields.
- ✅ **Gap 7 (lightweight)** → ray-attention v3 is small; avoid diffusion/GS unless absolutely necessary.
- ⚠️ **Gap 8 (domain agnostic)** → camera embedding is present, but domain-invariant training (gradient reversal or meta-learning) is not.
- ⚠️ **Gap 9 (multi-person)** → not currently in scope; could be added if needed but would widen the paper.
- ⚠️ **Gap 10 (robot motions)** → no AMASS/teleop data in workspace; synthetic AMASS generation is a future dependency.

---

## 5. Concrete recommendations for ICRA/CVPR 2027

1. **Run the cross-calibration ablation now.** Train `ray_attention_v3` on Shelf, evaluate on Campus and H36M; report MPJPE and reprojection error. This directly addresses Gap 1.
2. **Add a bone-length + temporal consistency loss** to the v3 trainer. This is the cheapest way to cover Gap 4 and begin closing Gap 5 without adding a full SMPL stage.
3. **Export uncertainty fields from `ray_attention_v3`.** Add per-view weights and per-joint reprojection residual to `HumanMotionIR`; this is the ICRA-relevant contribution (Gap 6).
4. **Keep the plugin interface minimal.** Do not chase diffusion/SSM unless v3 saturates; the robotics story is interpretable, lightweight fusion (Gap 2 & 7).
5. **Scope the paper around three claims:**
   - Geometry-aware learned fusion beats classical triangulation under occlusion/outlier views.
   - The same model generalizes across camera rigs without per-rig finetuning.
   - Uncertainty-aware output is useful for downstream robotics.

---

## 6. Risk / opportunity matrix

| Gap | Opportunity | Difficulty | Impact for paper |
|---|---|---|---|
| Cross-calibration generalization | Cross-dataset eval + domain-agnostic camera embedding | Medium | High |
| Outlier/occlusion robustness | Controlled real-data outlier benchmark | Low | High |
| Temporal/motion consistency | Bone-length + temporal loss in v3 trainer | Low | Medium |
| Parametric body recovery | Multi-view SMPL fit using v3 weights | High | High |
| Uncertainty for robotics | Extend `HumanMotionIR` fields | Low | Medium |
| Real-time / plug-in | Keep v3 lightweight; benchmark latency | Low | Medium |

---

## 7. References consulted

- Chharia et al., "MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation," CVPR 2025.
- Choi et al., "MVDoppler-Pose: Multi-Modal Multi-View mmWave Sensing for Long-Distance Self-Occluded Human Walking Pose Estimation," CVPR 2025.
- arXiv: DisPOSE (2026), RUMPL (2025), COMPOSE (2026), SkelSplat (2025), DenseWarper (2026), DeProPose (2025), RapidPoseTriangulation (2025), HeatFormer (2024).
- Project docs: `docs/design_v3.md`, `docs/swarm_iter3/cvpr2027_paper_positioning.md`, `docs/swarm_iter3/icra2027_paper_positioning.md`, `docs/swarm_iter4/cvpr_2027_positioning.md`, `docs/swarm_iter4/icra2027_positioning_experimental_plan.md`.
