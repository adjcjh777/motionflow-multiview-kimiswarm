# Swarm Iter-14 Synthesis: 20-Agent Proposal Review

Synthesis of the 20 proposals in `docs/swarm_iter14/`. No training was run; this document ranks the ideas by near-term feasibility and expected ROI for an ICRA/CVPR 2027 submission.

---

## 1. Proposal Inventory

| # | Proposal File | Title | Key Idea (1 sentence) |
|---|-------------|-------|----------------------|
| 1 | `proposal_webbridge-mpi-h36m-shelf-campus-blend.md` | WebBridge Multi-Dataset Mixing | Extend the mixed-dataset loader and per-dataset heads to include Shelf/Campus alongside MPI-INF-3DHP, H36M, and AIST++ for broader cross-dataset generalization. |
| 2 | `proposal_viewpoint-equivariant-fusion.md` | Viewpoint-Equivariant Fusion | Enforce SO(3) equivariance by rotating the whole camera rig during training and adding an explicit consistency loss, without adding new model parameters. |
| 3 | `proposal_skeleton-graph-residual-refinement.md` | Skeleton-Graph Residual Refinement | Replace the dense per-joint residual MLP with a small skeleton-graph message-passing module that propagates corrections along bone/symmetry edges. |
| 4 | `proposal_sota-comparison-harness-extension.md` | SOTA Comparison Harness Extension | Extend the SOTA comparison script with RANSAC-DLT/top-2 DLT baselines, richer per-joint/per-view metrics, latency, and parameter counts to produce paper-ready tables. |
| 5 | `proposal_ssl-masked-view-joint-prediction.md` | SSL Masked View/Joint Prediction | Extend the existing SSL pretext task to mask both views and joints, learning finer occlusion-completion cues before supervised fine-tuning. |
| 6 | `proposal_webbridge-target-domain-adaptation.md` | WebBridge Target Domain Adaptation | Warm-start the PP anchor and adapt it to a target dataset using the existing GRL+FiLM `DomainAdaptationWrapper` with a shared 17-joint skeleton. |
| 7 | `proposal_differentiable-bundle-adjustment-layer.md` | Differentiable Bundle Adjustment Layer | Add a differentiable Gauss-Newton/Levenberg-Marquardt bundle-adjustment layer after DLT to jointly refine 3D pose and camera parameters. |
| 8 | `proposal_learned-gauss-newton-triangulation.md` | Learned Gauss-Newton Triangulation | Replace the anchor's weighted-DLT step with DLT initialization + iterative Gauss-Newton refinement driven by the learned per-view weights. |
| 9 | `proposal_dynamic-view-selection-gate.md` | Dynamic View-Selection Gate | Insert a lightweight per-view/per-joint sigmoid gate before triangulation so the model learns to drop noisy/occluded views per joint. |
| 10 | `proposal_robustness-matrix-severity-sweeps.md` | Robustness Matrix Severity Sweeps | Replace the single-severity robustness matrix with severity sweeps and combined corruptions to identify exact failure thresholds. |
| 11 | `proposal_lightweight-factorized-st-attention-for-speed.md` | Lightweight Factorized ST Attention for Speed | Replace full transformer blocks in the factorized path with lightweight single-head attention + temporal pooling to recover accuracy and cut latency. |
| 12 | `proposal_occlusion-aware-visibility-v2-plus.md` | Occlusion-aware Visibility v2+ | Add a visibility-conditioned residual branch after DLT in the visibility-v2 model to compensate for occluded views. |
| 13 | `proposal_per-view-uncertainty-for-triangulation.md` | Per-View Uncertainty for Triangulation | Add a per-view/per-joint log-variance head and use predicted precision to re-weight DLT triangulation. |
| 14 | `proposal_epipolar-geometry-attention.md` | Epipolar Geometry Attention | Bias cross-view attention logits with epipolar-line distances so geometrically implausible matches are down-weighted. |
| 15 | `proposal_multiscale-temporal-clip-fusion.md` | Multi-Scale Temporal Clip Fusion | Fuse pose estimates from multiple temporal windows (short/medium/long) with a lightweight scale-mixing head. |
| 16 | `proposal_reprojection-consistency-loss.md` | Reprojection-Consistency Loss | Add a robust 2D reprojection loss on both raw and refined 3D poses using the corrected intrinsics from the PP head. |
| 17 | `proposal_neural-3d-pose-diffusion-prior.md` | Neural 3D Pose Diffusion Prior | Train a lightweight DDPM pose refiner on top of the frozen anchor output to suppress residual non-rigid errors. |
| 18 | `proposal_ensemble-factorized-temporal-anchors.md` | Ensemble of Factorized and Temporal Anchors | Late-fuse predictions from the temporal anchor, factorized anchor, and visibility-v2 anchor with learned weights or a residual meta-head. |
| 19 | `proposal_hierarchical-crossview-attention.md` | Hierarchical Cross-View Attention | Reorder the factorized attention into a dedicated view stage followed by a dedicated temporal stage instead of alternating layers. |
| 20 | `proposal_camera-parameter-conditioned-attention.md` | Camera-Parameter-Conditioned Attention | Encode K/R/t as per-view embeddings and inject them into cross-view attention as key/value/bias terms. |

---

## 2. Tier Ranking

### Tier 1: Run Next — Minimal Code, High ROI

These changes either reuse the existing 9.32 mm anchor directly or add a tiny module/loss, and can be smoke-tested in ≤5 epochs on a single RTX 4090.

| # | Proposal | Rationale |
|---|----------|-----------|
| 16 | Reprojection-Consistency Loss | Adds a loss term to the existing PP trainer (`experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`); no new model, directly targets the anchor's principal-point/focal robustness. Smoke: 5 epochs, ~12–18 min. |
| 2 | Viewpoint-Equivariant Fusion | No new model parameters; forks the existing PP trainer to add SO(3) rotation augmentation and an equivariance consistency term. Smoke: 5 epochs, ~20 min. |
| 13 | Per-View Uncertainty for Triangulation | Adds a single small log-variance head to the PP model; otherwise reuses the anchor backbone. Smoke: 5 epochs, ~8–12 min. |
| 9 | Dynamic View-Selection Gate | Adds a tiny 2-layer per-joint MLP gate (~3k params) inserted before triangulation. Smoke: 5 epochs, ~20–30 min. |
| 3 | Skeleton-Graph Residual Refinement | Single-module swap of the residual MLP for a skeleton-graph refiner; reuses existing `GraphJointRelation` builders. Smoke: 5 epochs, ~15–25 min. |

### Tier 2: Queue After Tier 1 — New Module, Low/Medium Risk

These add a modest new module or architecture branch but stay close to the existing anchor or its factorized/visibility variants.

| # | Proposal | Rationale |
|---|----------|-----------|
| 11 | Lightweight Factorized ST Attention for Speed | New lightweight factorized model, but builds on the existing factorized PP base; success would unblock a faster deployment variant. Smoke: 5 epochs, ~3–5 min. |
| 12 | Occlusion-aware Visibility v2+ | Extends the existing visibility-v2 model with a residual branch; medium complexity but well-scoped. Smoke: 5 epochs, ~10–15 min. |
| 14 | Epipolar Geometry Attention | New model subclass with geometry bias in cross-view attention; pure PyTorch geometry, no new trainable weights by default. Smoke: 5 epochs, ~25–35 min. |
| 19 | Hierarchical Cross-View Attention | Reorders factorized attention stages; minimal new code, but must beat the prior 57.68 mm factorized smoke. Smoke: 5 epochs, ~15–25 min. |
| 20 | Camera-Parameter-Conditioned Attention | New attention embedding module; small MLP for K/R/t encoding. Smoke: 3–5 epochs, ~8–12 min. |

### Tier 3: High Risk / High Reward — Major Architecture or Data Work

These require significant new architecture, data engineering, or training pipelines. Run only after Tier 1 and Tier 2 have delivered stable baselines and free GPU time.

| # | Proposal | Rationale |
|---|----------|-----------|
| 8 | Learned Gauss-Newton Triangulation | Replaces DLT with an iterative GN refinement; differentiable optimization carries NaN/instability risk but could improve triangulation. Smoke: 5 epochs, ~15–25 min. |
| 7 | Differentiable Bundle Adjustment Layer | Most complex: joint refinement of pose + cameras via autograd Jacobians; high implementation risk, high reward if it fixes calibration robustness. Smoke: 3–5 epochs, ~20–35 min. |
| 15 | Multi-Scale Temporal Clip Fusion | Processes three temporal window banks per clip; large memory/long-clip pipeline. Smoke: 3–5 epochs, ~25–30 min; full run ~18–24 h. |
| 17 | Neural 3D Pose Diffusion Prior | Post-hoc generative refiner; requires a separate training stage on frozen anchor outputs. Smoke: 5 epochs, ~5–10 min. |
| 1 | WebBridge Multi-Dataset Mixing | New data-loader integration for Shelf/Campus; risk of skeleton ordering mismatches and heterogeneous scale. Smoke: 3 epochs, ~5–10 min. |
| 6 | WebBridge Target Domain Adaptation | New joint-mapping + source/target mixing pipeline plus GRL+FiLM wrapper; depends on mixed canonical 17-joint loader. Smoke: 3 epochs, ~2–3 min. |
| 5 | SSL Masked View/Joint Prediction | Requires SSL pre-training then downstream fine-tuning; long pipeline, uncertain transfer gain. Smoke: 3 epochs synthetic, ~1–2 min. |

### Tier 4: Paper / Eval Infrastructure

These are evaluation and reporting improvements with no new training architecture. They can run concurrently with smoke training on CPU or A800-D read-only.

| # | Proposal | Rationale |
|---|----------|-----------|
| 4 | SOTA Comparison Harness Extension | Adds baselines (RANSAC-DLT, top-2 DLT), per-joint/per-view metrics, latency, and parameter counts; runs on existing checkpoints. |
| 10 | Robustness Matrix Severity Sweeps | Severity sweeps and combined corruptions on existing checkpoints; CPU/GPU eval only. |
| 18 | Ensemble of Factorized and Temporal Anchors | Technically a training proposal, but it is infrastructure-like because it only fuses already-trained checkpoints; should run after Tier 2 factorized and visibility-v2 jobs produce checkpoints. |

---

## 3. Recommended RTX 4090 GPU Queue Order

Single-GPU queue, respecting dependencies and smoke turnaround time.

1. **Tier 4 / eval baseline run** — `SOTA Comparison Harness Extension` and `Robustness Matrix Severity Sweeps` on the 9.32 mm anchor. These set the paper's comparison tables and can run overnight on CPU/A800 if the 4090 is needed for smoke training.
2. **Tier 1 block (parallelize in order of shortest runtime):**
   1. `Reprojection-Consistency Loss`
   2. `Per-View Uncertainty for Triangulation`
   3. `Viewpoint-Equivariant Fusion`
   4. `Dynamic View-Selection Gate`
   5. `Skeleton-Graph Residual Refinement`
3. **Tier 2 block:**
   1. `Lightweight Factorized ST Attention for Speed`
   2. `Camera-Parameter-Conditioned Attention`
   3. `Hierarchical Cross-View Attention`
   4. `Epipolar Geometry Attention`
   5. `Occlusion-aware Visibility v2+`
4. **Tier 3 block (after Tier 2 checkpoints exist):**
   1. `Learned Gauss-Newton Triangulation`
   2. `Differentiable Bundle Adjustment Layer`
   3. `WebBridge Target Domain Adaptation` (depends on mixed 17-joint loader)
   4. `WebBridge Multi-Dataset Mixing`
   5. `Multi-Scale Temporal Clip Fusion`
   6. `Neural 3D Pose Diffusion Prior` (uses frozen anchor outputs; can run when 4090 is free)
   7. `SSL Masked View/Joint Prediction` (long pre-training + fine-tune pipeline)
5. **Final integration:**
   1. `Ensemble of Factorized and Temporal Anchors` (requires Tier 2 factorized and visibility-v2 checkpoints).

*Estimated active RTX 4090 time to clear all smokes: ~6–10 hours if run serially; many Tier 1/Tier 2 smokes are short enough to batch in a single day.*

---

## 4. CPU / A800-D Read-Only Recommendations

The following proposals need only existing checkpoints and can run on CPU or on A800-D in read-only mode:

- **`proposal_sota-comparison-harness-extension.md`**: Pure eval/reporting; DLT/IRLS baselines are CPU-only; learned anchor inference can run on A800-D read-only.
- **`proposal_robustness-matrix-severity-sweeps.md`**: Pure evaluation of existing checkpoints; full sweep is CPU/GPU optional.
- **`proposal_ensemble-factorized-temporal-anchors.md`**: If constituent anchor predictions are cached to `.npy` first, the ensemble head can be trained on CPU; eval is inference-only.
- **`proposal_neural-3d-pose-diffusion-prior.md`**: Training the lightweight refiner is cheap GPU, but generating frozen anchor outputs and metric computation are CPU-friendly; full eval can run on A800-D read-only.

All other proposals require a training smoke on the local RTX 4090 because they introduce new model parameters or loss terms that must be validated with gradients before CPU/A800 integration.

---

## 5. Self-Evolution Loop Mapping

For every proposal, the loop is:

1. **Reflect:** Review the 9.32 mm anchor (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`) and the iter13 robustness matrix (`docs/results_iter13.md`, `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py`) to identify the weakest axis (calibration robustness, occlusion, latency, triangulation, data coverage, or evaluation depth).
2. **Hypothesize:** Propose a minimal, architecture-preserving change (new loss, small head, new module, or eval harness) that addresses the weak axis while reusing the anchor backbone.
3. **Smoke-validate:** Run ≤5 epochs on a tiny subset (typically 500 samples, `d=32`, `residual_hidden=64`) on the RTX 4090; check for NaNs, finite val MPJPE, and that the change does not regress the anchor by more than the proposal's stated threshold.
4. **Integrate:** If the smoke passes, register the new model/loss in `motionflow_mv/fusion/__init__.py` or `experiments/eval_full_metrics.py`, add the eval script to the paper harness, and queue the full run; if it fails, apply the fallback and either re-smoke or archive the proposal.

---

## 6. Bottom-Line Recommendation

Start with the **Tier 1** block: `Reprojection-Consistency Loss`, `Per-View Uncertainty`, `Viewpoint-Equivariant Fusion`, `Dynamic View-Selection Gate`, and `Skeleton-Graph Residual Refinement`. These five can all be smoke-tested within one day on the RTX 4090 and are the most likely to improve or preserve the 9.32 mm anchor while generating paper-ready ablations.

Run the **Tier 4** eval infrastructure (`SOTA Comparison Harness Extension`, `Robustness Matrix Severity Sweeps`) in parallel on CPU/A800-D read-only so the paper tables and robustness curves are ready before the new training variants finish.

Defer **Tier 3** high-risk items until at least two Tier 2 architectures have produced stable checkpoints, because several Tier 3 ideas (domain adaptation, multi-dataset mixing, ensemble) depend on those checkpoints or on a validated mixed-data pipeline.
