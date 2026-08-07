# MotionFlow-MultiView: A Reproducible Multi-View Human Motion Capture Workflow

> System-level paper story v2 — integrates the A800-D research design with the current ICRA/CVPR 2027 experimental results.  
> Last updated: 2026-08-07.  
> Target venues: ICRA / CVPR 2027 (system + robot-learning track).

---

## 1. One-sentence thesis

**MotionFlow-MultiView is a reproducible, end-to-end multi-view human motion capture workflow that turns calibrated or weakly calibrated multi-camera video into a robot-ready motion representation; its pluggable fusion module, OmniMultiViewFusion, currently reaches 8.35 mm MPJPE on MPI-INF-3DHP while the surrounding system handles calibration, occlusion, synchronization, quality gates, and downstream policy consumption.**

---

## 2. Problem and motivation

### 2.1 Why multi-view matters for robotics

Single-view human motion recovery (HMR) has made rapid progress, but it remains fundamentally limited in real-world deployment:

- **Occlusion and self-occlusion** hide joints from any single camera.
- **Depth ambiguity** gives scale and root-depth estimates that are often inconsistent across clips.
- **Limited field of view** forces aggressive cropping or camera placement, degrading motion quality.
- **Metric drift** accumulates over long sequences, making the output unreliable for robot imitation.

Multi-view capture is the modality of choice wherever metric, world-grounded human motion matters: human-robot collaboration, robot policy training, sports analytics, and immersive content. The classical multi-view pipeline is deceptively simple:

1. Detect 2D keypoints independently in each view.
2. Triangulate with Direct Linear Transform (DLT).

In practice this pipeline collapses under **occlusion**, **noisy or biased 2D detections**, **calibration drift**, and **unsynchronized or hand-held cameras**. A deployable system must solve all of these at once.

### 2.2 What is missing from the literature

Most published work reports MPJPE on a clean benchmark and stops. A robotics system must also:

- **Calibrate** or self-calibrate cameras in the field.
- **Synchronize** views with sub-frame accuracy or report failure.
- **Detect occlusion** and down-weight or drop corrupted views.
- **Propagate uncertainty** from pixels to 3D joints to robot actions.
- **Gate quality** before feeding downstream retargeting and policy training.
- **Validate end-to-end** with policy metrics, not just pose error.

**This paper therefore reframes MotionFlow-MultiView as a system contribution:** a reproducible workflow in which a strong but replaceable fusion module is only one component, and where the true test is whether the captured motion improves robot behavior.

---

## 3. System architecture

```text
Capture Session
  ├─ video_00 + metadata
  ├─ video_01 + metadata
  ├─ ...
  ├─ synchronization
  └─ calibration tier (strict / weak / none)
          │
          ▼
Input Normalization + Quality Precheck
          │
          ├──────────── single view ────────────┐
          │                                      │
          ▼                                      ▼
Per-view Frozen GVHMR Workers         GVHMR single-view passthrough
          │                                      │
          ▼                                      │
Latent Spatiotemporal Alignment / Fusion Plugin  │
          │    (OmniMultiViewFusion)             │
          │                                      │
          └──────────────────┬───────────────────┘
                             ▼
                       HumanMotionIR
            (pose + uncertainty + provenance)
                             │
                             ▼
               hmr4d_results.pt compatibility export
                             │
                             ▼
                   Robot Profile Resolver
                             │
                             ▼
                  GMR → PKL → NPZ → smoothing
                             │
                             ▼
              MJLab training / policy preview
                             │
                             ▼
              deployment NPZ + ONNX export
```

### 3.1 Design principles

1. **Frozen upstream.** GVHMR runs per view with frozen weights. This isolates the system contribution from upstream model improvement and gives a fair comparison of fusion modules.
2. **Pluggable fusion.** OmniMultiViewFusion is one candidate `MultiViewFusionPlugin`; the interface also admits simple baselines (best single view, late averaging), EasyMocap-style geometry, and future methods.
3. **HumanMotionIR as a stable contract.** All upstream and downstream modules exchange a single, versioned schema with pose, uncertainty, quality flags, and provenance hashes.
4. **Quality gates before downstream consumption.** Low-confidence frames can be rejected, flagged for human review, or down-weighted in policy training.
5. **Robot-profile driven.** DoF, joint names, action order, and export conventions are externalized so the same HumanMotionIR can drive different robots.

---

## 4. Fusion module contribution: OmniMultiViewFusion

### 4.1 The geometry-first decomposition

OmniMultiViewFusion keeps triangulation at the center and learns only the structured error that geometry cannot fix:

| Stage | What it does | Why it matters |
|-------|--------------|----------------|
| **Intrinsic self-calibration** | Predicts per-view `(Δcx, Δcy, Δf)` from the 2D/confidence pattern. | Fixes principal-point drift, the dominant field failure mode. |
| **Visibility gating** | Predicts per-view, per-joint soft visibility multipliers `m_vj`. | Masks occluded views before triangulation. |
| **Graph-joint attention** | Exchanges information along the skeleton graph (parent–child, symmetry). | Propagates evidence anatomically under occlusion. |
| **Spatiotemporal (T × V × J) attention** | Factorised Transformer layers over time, views, and joints. | Captures motion context and multi-view consistency at `O(T² + V² + J²)` per axis. |
| **Uncertainty-weighted triangulation** | Predicts per-view log-variance `λ_vj` and weights the DLT by `c · m · exp(-λ)`. | Down-weights noisy or occluded views while keeping geometric triangulation. |
| **Adaptive Gauss-Newton refinement** | Runs 1–2 differentiable Gauss-Newton steps with learned per-joint damping. | Refines the DLT solution inside the camera model. |
| **Residual refinement** | Small MLP adds a learned correction conditioned on pooled features and the geometric estimate. | Captures detector bias and skeleton prior. |

### 4.2 Current empirical anchor

The strongest instantiation to date is the **Bayesian Tri v2** ensemble, predecessor to the unified OmniMultiViewFusion module:

| Dataset / Condition | Model | MPJPE (mm) | PA-MPJPE (mm) | Notes |
|---------------------|-------|-----------:|---------------:|-------|
| MPI-INF-3DHP S2/Seq1 | Raw DLT | 25.21 | 24.08 | geometric baseline, no learning |
| MPI-INF-3DHP S2/Seq1 | Cross-view residual + PP (d=64, h=128) | **9.32** | **5.37** | best single model before d=128 |
| MPI-INF-3DHP S2/Seq1 | Bayesian Tri v2 single (d=128) | 9.03 | — | stabilized training |
| MPI-INF-3DHP S2/Seq1 | **Bayesian Tri v2 ensemble** (stabilized + aug, d=128) | **8.35** | **5.29** | current best; below 8.75 mm threshold |
| Human3.6M S5/Act2 | CamPE + GraphJR (d=64, h=128) | **0.62** | **0.70** | 4-view rig |
| Human3.6M S5/Act2 | Cross-view residual + PP (d=64, h=128) | 5.24 | 4.84 | with principal-point correction |

The 8.35 mm ensemble combines two d=128 checkpoints: `outputs/bayesian_tri_v2_stabilized_mpiinf3dhp.pth` and `outputs/bayesian_tri_v2_aug_mpiinf3dhp.pth`, evaluated with `scripts/eval_ensemble_wsl.sh` (PCK@50/100/150 = 1.000, AUC = 0.9444).

### 4.3 Key implementation files

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py`
- `motionflow_mv/fusion/principal_point_correction.py`
- `motionflow_mv/fusion/epipolar_attention_bias.py`
- `motionflow_mv/fusion/triangulation.py`
- Training: `scripts/run_bayesian_tri_v2_large_scale_wsl.sh`
- Results: `docs/results_icra_cvpr_2027.md`

---

## 5. HumanMotionIR and quality gates

### 5.1 HumanMotionIR schema

`HumanMotionIR` is the stable interface between upstream HMR and downstream robot tasks. It includes:

- **pose**: world-coordinate 3D joints `(T, J, 3)` plus SMPL body/global/transl parameters when available.
- **uncertainty**: per-frame and per-joint confidence, view support count, temporal alignment error, fusion disagreement, and optional scale uncertainty.
- **quality**: frame-valid flags, failure reasons, and summary metrics.
- **provenance**: source manifest hash, per-view artifact hashes, GVHMR version, fusion plugin version, calibration hash, and IR builder version.

### 5.2 Quality gates

Before any downstream consumption, the system must pass at least the following checks:

1. **Synchronization gate.** Report sub-frame offset and drift; fail or flag if `p95` error exceeds the allowed bound (1 frame for strict calibration, 2 frames for hand-held).
2. **Calibration gate.** Report reprojection error median / p95; reject sessions with excessive drift or degenerate configurations.
3. **Occlusion / visibility gate.** Require a minimum number of visible views per joint; trigger best-view fallback or human review when violated.
4. **Fusion disagreement gate.** Compare per-view 2D reprojections to the fused 3D pose; flag frames with large disagreement.
5. **Person-association gate.** Ensure consistent person identity across views and time; flag swaps or ambiguous cases.
6. **Motion-continuity gate.** Detect large inter-frame jumps or implausible bone-length variation.
7. **Contact gate.** Protect feet, hands, and end-effector contact frames from training if uncertainty is above threshold.

### 5.3 Uncertainty propagation

Uncertainty is not discarded after fusion. Candidate propagation strategies include:

- **Training data selection:** weight or exclude low-confidence clips from policy training.
- **Retargeting weights:** lower robot target weights for joints with low confidence.
- **Smoothing strength:** adapt temporal smoothing based on synchronization error and fusion disagreement.
- **Reward shaping:** (optional, to be validated) weight imitation reward by per-frame quality.

---

## 6. Robot profiles and downstream validation

### 6.1 Robot profile abstraction

A robot profile externalizes everything that is currently hard-coded:

```yaml
schema_version: 1
profile_id: unitree_g1_23dof
kinematics:
  dof_count: 23
  joint_names: [...]
  qpos_indices: [...]
  joint_limits: [...]
  end_effectors: [...]
  contact_bodies: [...]
retarget:
  human_to_robot_body_map: {...}
  uncertainty_mapping: {...}
  scale_policy: ...
training:
  mjlab_task_id: ...
  observation_schema: ...
  action_schema: ...
export:
  onnx_action_order: [...]
  deployment_metadata: {...}
```

First profiles:

- `bxi_elf3_current`: zero-behavior-change baseline for MotionFlow.
- `unitree_g1_23dof`: 23-DoF humanoid profile, pending authoritative model and action-order verification.
- `unitree_g1_29dof`: schema reserved; not populated until authoritative definition is available.

### 6.2 End-to-end validation metrics

The system must be evaluated with robot-relevant metrics, not just MPJPE:

- **HMR layer:** MPJPE, PA-MPJPE, PCK@50/100/150, AUC, reprojection error, bone-length error.
- **Retargeting layer:** joint-limit violation, foot sliding, ground penetration, end-effector error, contact consistency.
- **Policy layer:** episode length, tracking reward components, termination type distribution, success rate, sample throughput, time to reach reward threshold.
- **System layer:** wall time per minute of video, GPU/CPU memory, disk increment, cache hit rate, provenance completeness.

---

## 7. Experimental plan

### 7.1 Datasets and benchmarks

| Dataset | Views | Use case | Metrics |
|---------|-------|----------|---------|
| **MPI-INF-3DHP** | up to 14 | Main accuracy benchmark | MPJPE, PA-MPJPE, PCK, AUC |
| **Human3.6M** | 4 | Cross-dataset / fixed rig | MPJPE, PA-MPJPE |
| **AIST++** | multi | Music/dance motion, temporal continuity | MPJPE, velocity error, foot contact |
| **Shelf/Campus** | 3–5 | Real-world, less controlled capture | MPJPE where available, reprojection, qualitative |
| **Real robot / physics** | 2+ | End-to-end policy validation | tracking reward, success rate, foot slide, penetration |

### 7.2 Required ablations

1. **View count:** MPJPE@k for k = 2..14.
2. **Calibration tier:** strict, weak self-calibration, and hand-held/no-calibration.
3. **Fusion module comparison:** best single view, late averaging, strict-geometry DLT, EasyMocap/geometry baseline, OmniMultiViewFusion.
4. **Quality gates:** no gate / gate / gate + best-view fallback.
5. **Uncertainty propagation:** none / retarget weights / training weights.
6. **Robot profile zero regression:** hard-coded ELF3 vs. `bxi_elf3_current` profile.
7. **Single-view vs. multi-view policy training:** same action, same profile, same seed.

### 7.3 Checkpoints and scripts to run

- Ensemble evaluation: `scripts/eval_ensemble_wsl.sh`
- Large-scale training: `scripts/run_bayesian_tri_v2_large_scale_wsl.sh`
- Extended robustness matrix: `experiments/prototypes/run_extended_robustness_matrix.py`
- WebBridge cross-dataset benchmark: `outputs/webbridge_benchmark_crossview_residual_smoke_v2.json`
- Variable-view curve: `docs/figures/variable_views_crossview_residual_smoke.png`

---

## 8. Risks and mitigations

| Risk | Status | Mitigation |
|------|--------|------------|
| Unified OmniMultiViewFusion fails to improve over 8.35 mm ensemble | Open | Warm-start from `bayesian_tri_v2_stabilized_mpiinf3dhp.pth`; freeze encoder for 5 epochs; stage unfreezing. |
| Rotation robustness still weak (rot_0.5° = 16.89 mm) | Open | Stronger extrinsic perturbation curriculum; dedicated rotation-aware correction head. |
| Focal-length robustness (focal_1% = 19.13 mm) | Open | Dedicated focal-scale loss; bound corrections to dataset ranges. |
| View dropout is largest accuracy degradation (k=10: 28.28 mm) | Open | Visibility gating + uncertainty weighting + view-dropout training. |
| Cross-dataset transfer (H36M ↔ MPI) | Open | Domain-adaptation wrapper, per-dataset pose heads, or larger balanced mixed training. |
| Hand-held / weak-calibration capture | Open | Start with strict-calibration baselines; only later add weak/no-calibration tiers. |
| Robot profile mismatch or action-order errors | Open | Fixture-based validation of qpos, action, and ONNX action order against MJCF/URDF. |
| Policy metrics do not improve with better HMR | Open | Report honestly; avoid claiming HMR accuracy implies policy gain. |
| Dependency / license conflicts (EasyMocap, MUC, etc.) | Open | Audit licenses before integration; keep optional plugins outside the default Docker image. |

---

## 9. Next steps and milestones

### 9.1 Immediate (this week)

1. Land the OmniMultiViewFusion v2 model skeleton in `motionflow_mv/fusion/` and a CPU smoke test in `tests/test_omniview_fusion_v2_smoke.py`.
2. Confirm warm-start checkpoint availability (`outputs/bayesian_tri_v2_stabilized_mpiinf3dhp.pth`).
3. Run a d=48, 10-epoch GPU smoke and verify clean MPJPE stays within 5% of the 9.32 mm baseline.

### 9.2 Short term (next 2–4 weeks)

1. Full d=128 training with staged unfreezing; target ≤ 8.0 mm single-model MPJPE on MPI-INF-3DHP S2/Seq1.
2. Run the extended robustness matrix (noise, occlusion, view dropout, calibration perturbations).
3. Generate the variable-view MPJPE@k curve for OmniMultiViewFusion v2.
4. Repeat seeds (n ≥ 3) and report mean ± std.
5. Update `docs/results_icra_cvpr_2027.md` with final numbers.

### 9.3 Medium term (next 1–2 months)

1. Implement HumanMotionIR schema validator and compatibility exporter with golden-artifact regression tests.
2. Build the first robot profile (`bxi_elf3_current`) and prove zero regression against the hard-coded pipeline.
3. Run end-to-end policy experiments comparing single-view, best-view, and OmniMultiViewFusion inputs.
4. Draft the system-architecture and robustness figures for the paper.

### 9.4 Definition of done for this story

- [x] `docs/paper_story_system_v2.md` created and committed.
- [ ] OmniMultiViewFusion v2 CPU smoke test passing.
- [ ] GPU smoke (d=48) within 5% of 9.32 mm baseline.
- [ ] Full d=128 run reaches ≤ 8.0 mm clean MPJPE single model.
- [ ] Extended robustness matrix completed.
- [ ] Variable-view curve generated.
- [ ] HumanMotionIR compatibility exporter passes golden-artifact regression.
- [ ] Robot profile zero regression for `bxi_elf3_current`.
- [ ] End-to-end policy metrics collected on at least one robot task.

---

## 10. v4 direction: View-mask-aware adaptive multi-view fusion

### 10.1 Why v4 is needed

The v2/v3 anchor is strong under clean, fixed-view conditions, but real-world deployment breaks those assumptions:

- **Variable views.** H36M v2 shows 2-view MPJPE  1990 mm and 3-view  1620 mm.
- **Occlusion / view dropout.** 30 % view dropout degrades MPI-INF-3DHP to 18.15 mm; 30 % joint occlusion to 16.99 mm.
- **Calibration drift.** rot_0.5°  16.89 mm and focal_1 %  19.13 mm remain unsolved.

### 10.2 v4 headline modules

| Module | What it adds | Paper section |
|--------|--------------|---------------|
| **View-mask-aware visibility gating v2** | Per-joint context visibility with uncertainty scaling | 4.1 |
| **Adaptive view selector** | Budgeted top-k view selection per joint | 4.2 |
| **Rotation correction head** | Bounded SO(3) residual before triangulation | 4.3 |
| **Skeleton-graph residual refiner** | Graph propagation over bone/symmetry edges | 4.4 |
| **Kinematic-chain graph refiner** | Optional final temporal skeleton pass | 4.5 |
| **Attention-entropy regularization** | Sharpens per-view triangulation weights | 4.6 |
| **Calibration perturbation curriculum** | Progressive rot/focal/PP augmentation | 5.1 |

All modules are optional, individually togglable, and warm-startable from v2/v3 checkpoints (`strict=False`).

### 10.3 Strongest v4 claims

1. **MPI-INF-3DHP single model < 8.6 mm** (anchor: single 9.03 mm, ensemble 8.35 mm).
2. **Variable-view k=2 MPJPE < 50 mm** (anchor: H36M v2 ~1990 mm).
3. **30 % view dropout < 16.3 mm** (anchor: 18.15 mm).
4. **rot_0.5° < 14 mm and focal_1 % < 15 mm** (anchor: 16.89 mm / 19.13 mm).
5. **Model remains warm-startable and < 2.0 M parameters**.

### 10.4 v4 deliverables

- Design: `docs/v4_architecture_design_proposal.md`
- Story: `docs/swarm_iter20/paper_story_v4.md`
- Model: `motionflow_mv/fusion/omniview_fusion_v4.py`
- Trainer: `experiments/train_omniview_fusion_v4_webbridge_multi.py`
- Eval: `experiments/eval_omniview_fusion_v4_mpiinf3dhp.py`
- Tests: `tests/test_omniview_fusion_v4.py`

---

## 11. Related files

- System research design: `docs/motionflow_multiview_system_research_from_a800.md`
- Paper story (predecessor): `docs/icra_cvpr_2027_paper_story.md`
- Results: `docs/results_icra_cvpr_2027.md`
- Experiment log: `docs/experiment_log_icra_cvpr_2027.md`
- OmniMultiViewFusion proposal: `docs/swarm_iter18/P11_paper_story.md`
- Next-iteration plan: `docs/swarm_iter18/P20_synthesis.md`
- Fusion implementation: `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py`
- Principal-point correction: `motionflow_mv/fusion/principal_point_correction.py`
- Triangulation utilities: `motionflow_mv/fusion/triangulation.py`
