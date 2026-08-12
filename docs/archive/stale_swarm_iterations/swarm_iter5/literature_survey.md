# Literature Survey: Self-Evolution + Multi-View Human Pose (CVPR/ICRA 2025-2026)

**Date:** 2026-08-04
**Scope:** Bridge recent "self-evolution" style training ideas (LLM-style iterative self-improvement) with the 2025-2026 multi-view human pose literature, then extract 3-5 concrete ideas for the motionflow-multiview ICRA/CVPR 2027 push.

**Sources:** Project prior art in `docs/swarm_iter5/literature_gap.md`, `docs/swarm_iter5/literature_gap_summary.md`, `docs/swarm_iter4/cvpr_2027_positioning.md`, and the current `motionflow_mv/fusion/ray_attention_temporal_model.py` baseline. External arXiv queries were attempted but blocked by the local network, so this note uses the project’s already-curated references and the model’s current architecture as the working context.

---

## 1. What is "Qwen3.8-style self-evolution" in this context?

Recent large-model training (Qwen, DeepSeek-R1, etc.) shows that a model can be improved iteratively by:

1. **Generating its own training data** from a strong teacher or from sampling.
2. **Rewarding correctness** through an explicit, often rule-based reward (reward model, verifier, or metric).
3. **Iterative distillation / rejection sampling:** keep only the high-quality self-generated outputs, fine-tune, repeat.
4. **Self-critique / revision:** a separate (or shared) critic scores the output, proposes corrections, and the model is trained to revise.
5. **Curriculum growth:** start with easy samples, progressively increase difficulty/occlusion/noise.

For multi-view 3D pose, the analog is a **geometry-aware self-improving loop** where the model predicts 3D poses, the system measures consistency via reprojection / bone-length / temporal smoothness, and the model is refined on the corrected outputs. The project already has the ingredients: differentiable weighted DLT, per-view weights, temporal attention, and synthetic augmentation. The literature survey below turns these ingredients into concrete directions.

---

## 2. Recent multi-view pose literature (2025-2026) — project-aligned summary

Key papers already identified in the project’s literature gap analysis:

| Paper / Venue | Relevant finding for this project |
|---|---|
| **MV-SSM** (Chharia et al., CVPR 2025) | State-space view fusion; confirms structured fusion beyond vanilla attention. |
| **RUMPL** (arXiv 2025) | Ray-based transformer lifting; closest spirit to our `ray_attention` family. |
| **DisPOSE** (arXiv 2026) | Diffusion-based self-supervised multi-view pose; trades compute for label efficiency. |
| **COMPOSE** (arXiv 2026) | Hypergraph cover optimization over 2D detections; strong geometric prior, but slow. |
| **RapidPoseTriangulation / DeProPose** (arXiv 2025) | Fast adaptive fusion and triangulation; speed-accuracy tradeoff is relevant. |
| **From Sparse to Dense / DenseWarper** (arXiv 2026) | Spatio-temporal dense warping; temporal + dense correspondence is a frontier. |
| **Unconstrained Multi-view Human Pose with Algebraic Priors** (arXiv 2026) | Calibration-free pose from algebraic constraints; less relevant for calibrated robotics. |

**Main trend:** the field has shifted from "flatten cameras, regress 3D" to **geometry-aware learned fusion** (ray/camera embeddings + differentiable triangulation). The remaining differentiators for a 2027 paper are **temporal consistency, cross-calibration generalization, explicit uncertainty, and self-improving training** — all of which are under-represented in the 2025-2026 papers above.

---

## 3. Five concrete ideas for motionflow-multiview

### Idea 1 — Geometry-verified self-supervised fine-tuning ("pose-RFT")

**Concept:** Train the temporal ray-attention model in a self-evolution loop similar to Qwen’s rejection-sampled fine-tuning. After an initial supervised run on H36M / MPI-INF-3DHP, generate pseudo-labels on unlabeled in-the-wild multi-view clips, then use **geometric consistency as a verifier** to keep only the high-quality predictions.

**Verifier stack:**
- Reprojection error `< ε` pixels.
- Bone-length variance across the clip `< δ` (temporal skeleton consistency).
- Per-view weight entropy: not all joints can be dominated by one view.
- Temporal smoothness: 3D acceleration magnitude below a threshold.

**Why it fits the project:** `RayAttentionFusionModelTemporal` already predicts per-view weights and 3D joints; adding a geometric verifier requires no architecture change, only a data-selection loop.

**First experiment:**
- Train the existing `experiments/train_ray_attention_temporal_mpiinf3dhp.py` for 2 epochs (smoke run already at 25.25 mm).
- Generate pseudo-3D labels on Shelf/Campus videos.
- Filter by reprojection error and bone-length consistency, then fine-tune for 3-5 epochs.

---

### Idea 2 — Reward-weighted attention from reprojection ("ray-RLHF")

**Concept:** Treat the per-view attention weights as a policy and train them with a **reward signal derived from triangulation quality**. The reward is high when the weighted DLT output reprojects well to all views and low when occluded/outlier views receive too much weight.

**Implementation sketch:**
- Keep the transformer backbone.
- Add a small reward head `r = f(weights, reprojection_error, bone_length_error)`.
- Fine-tune the weight predictor with a reward-weighted KL objective: increase probability of weight assignments that yield low reprojection error.

**Why it fits the project:** The current `_extract_frame_features` + `weight_head` architecture is already a differentiable per-view weight generator. A reward-weighted fine-tuning step adds robustness to synthetic-to-real gaps and outlier views without growing the model size.

**First experiment:**
- Use the current MPI-INF-3DHP checkpoint as the base.
- Add a small `reward_head` and fine-tune for ≤10 epochs with a reward-weighted MSE/KL loss.
- Compare MPJPE before/after on the H36M cross-subject val set.

---

### Idea 3 — Self-critique temporal refiner ("pose-critic" branch)

**Concept:** Add a lightweight critic network that, given the model’s own 3D prediction and per-view weights, proposes **temporal corrections**. This mirrors LLM self-critique: predict → critique → refine.

**Architecture:**
- Predictor: existing temporal ray-attention model.
- Critic: a small 1D temporal convnet or transformer over the `(B, T, J, 3)` trajectory plus per-view weight entropy.
- Refiner: residual correction `X_refined = X_pred + α · critic(X_pred, weights)`.

**Loss:** train the refiner end-to-end with `L_MPJPE + λ_temporal · L_smooth + λ_bone · L_bone_length`.

**Why it fits the project:** `RayAttentionFusionModelTemporal` returns exactly `(B, T, J, 3)` and `(B, T, V, J)` weights. A critic operating on this output keeps the base model frozen during inference and only adds a small post-hoc refinement, which is attractive for ICRA-style plug-in deployment.

---

### Idea 4 — Curriculum self-evolution over occlusion and camera geometry

**Concept:** Build a curriculum that progressively exposes the model to harder multi-view scenarios. This is the CV analog of Qwen-style curriculum RL: easy synthetic clips first, then harder real-world clips with motion blur, occlusion, and calibration drift.

**Curriculum dimensions:**
1. **Occlusion level:** start with 0% joint occlusion, increase to 30-50%.
2. **Camera count:** start with 4 views, then 3, then 2 (few-view pose).
3. **Camera geometry drift:** add small per-frame perturbations to `K, R, t` to simulate recalibration errors.
4. **Motion complexity:** start with slow walking, add sports/fast gestures.

**Why it fits the project:** The current `augment_clip` already injects noise, dropout, and outliers. A curriculum wrapper around `TemporalClipDataset` is minimal new code.

**First experiment:**
- Add a `CurriculumSampler` to `experiments/train_ray_attention_temporal_mpiinf3dhp.py` that increases dropout and occlusion probability linearly over the first half of training.
- Report whether final val MPJPE improves over the fixed-augmentation baseline.

---

### Idea 5 — Cross-dataset self-distillation for plug-in generalization

**Concept:** Use the model trained on one camera rig (e.g., MPI-INF-3DHP) as a teacher to generate pseudo-labels for a different rig (Shelf/Campus), then distill a **smaller student** that is robust to camera geometry changes. This directly addresses the cross-calibration generalization gap identified in the project’s gap analysis.

**Pipeline:**
1. Teacher: `RayAttentionFusionModelTemporal` trained on source rig with 3D GT.
2. Generate pseudo-3D labels for target rig videos.
3. Student: same architecture but with `d=32` and `n_temporal_layers=1`.
4. Distill with `L_MPJPE(student, teacher) + λ_camera · L_camera-embedding-alignment`.

**Why it fits the project:** The camera-conditioned embedding already encodes `K, R, t`; a distillation loss that aligns embeddings across rigs can enforce geometry invariance. This makes the final plugin portable without per-rig fine-tuning — a strong ICRA claim.

---

## 4. Prioritization

| Idea | New code needed | GPU cost (smoke) | Expected impact | Risk |
|---|---|---|---|---|
| 1. Geometry-verified self-supervised fine-tuning | Data-selection loop only | Low (reuse checkpoint) | Medium | Pseudo-label quality depends on base model. |
| 2. Reward-weighted attention | Reward head + loss | ≤10 epochs | Medium | Hyper-sensitive to reward scale. |
| 3. Self-critique temporal refiner | Critic + refiner network | ≤10 epochs | High | Adds parameters; needs careful tuning. |
| 4. Curriculum over occlusion/cameras | Sampler wrapper only | ≤10 epochs | Medium | May need more epochs to show benefit. |
| 5. Cross-dataset self-distillation | Teacher/student scripts | ≤10 epochs (student) | High | Requires target-rig multi-view data. |

**Recommended path for the next 1-2 weeks:** implement **Idea 3** (self-critique refiner) and **Idea 4** (curriculum sampler) because both are small, self-contained extensions of the existing temporal model and directly strengthen the paper’s empirical story. Then add **Idea 5** once Shelf/Campus data is available.

---

## 5. References

- Project references: `docs/swarm_iter5/literature_gap.md`, `docs/swarm_iter5/literature_gap_summary.md`, `docs/swarm_iter4/cvpr_2027_positioning.md`.
- Chharia et al., “MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation,” CVPR 2025.
- Choi et al., “MVDoppler-Pose: Multi-Modal Multi-View mmWave Sensing for Long-Distance Self-Occluded Human Walking Pose Estimation,” CVPR 2025.
- RUMPL, DisPOSE, COMPOSE, SkelSplat, DenseWarper, DeProPose, RapidPoseTriangulation: arXiv 2025-2026 pre-prints (cited in `docs/swarm_iter5/literature_gap.md`).
- Self-evolution / RLHF analogy: Qwen / DeepSeek-R1 style iterative reward-based refinement (general methodology, no project dependency).

---

## 6. Notes and blockers

- **External literature fetch failed.** `curl` to `export.arxiv.org` returned an empty reply (see `tmp/` network log if needed). The survey therefore relies on the project’s existing curated references and the current model implementation.
- **No existing code was modified.** The ideas above are designed to be implemented as new scripts (e.g., `train_ray_attention_temporal_reward_v*.py`, `curriculum_sampler_v*.py`) to avoid touching the working baseline.
- **GPU training note:** All proposed first experiments are designed for ≤10 epochs / ≤30 min on the RTX 4090, matching the project’s smoke-run policy.
