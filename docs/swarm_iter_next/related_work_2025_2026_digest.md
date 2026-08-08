# Multi-View Pose Literature Digest: 2025-2026

> Project baseline: **20.24 mm val_MPJPE (v18 baseline)**  
> Purpose: Identify the most actionable 2025-2026 directions for the next iteration (v26).

## 1. TL;DR for v26

The strongest near-term opportunities for the next iteration are:

1. **Spatio-temporal warping before triangulation** (DenseWarper, 2026). The v26 proposal already plans deformable spatio-temporal cross-view attention; DenseWarper supports the value of warping dense temporal features rather than attending over the full `(T*V)^2` tensor.

2. **State-space cross-view blocks** (MV-SSM, CVPR 2025). If transformer attention remains view-count sensitive, a Mamba-style drop-in replacement is now well motivated.

3. **Adaptive deficiency weighting** (DeProPose, 2025). Should be folded into v26's temporal aggregation so noisy/occluded frames are down-weighted.

4. **Differentiable rendering / generative priors** (SkelSplat, DisPOSE, 2026). Longer-term regularisers for neural BA or multi-person extension.

## 2. Papers by Theme

### Temporal / Spatio-Temporal Fusion

- **MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation** (CVPR 2025)  
  Link: <https://arxiv.org/abs/2509.00649>  
  *Key idea:* Replace vanilla cross-view attention with a Projective State Space (PSS) block and Grid Token-guided Bidirectional Scan (GTBS) to preserve spatial structure and generalise across camera arrangements.  
  *Relevance:* Strongest temporal/structural alternative to our transformer fusion. Supports variable-view inference and could inform a v26 state-space cross-view aggregator.

- **DeProPose: Deficiency-Proof 3D Human Pose Estimation via Adaptive Multi-View Fusion** (arXiv 2025)  
  Link: <https://arxiv.org/abs/2502.16419>  
  *Key idea:* Adaptive multi-view fusion using relative projection error to weight noisy, occluded, or missing views.  
  *Relevance:* Directly maps to our extended robustness matrix. Reinforces the need for adaptive view weighting in v18/v23 and future v26 temporal fusion.

- **From Sparse to Dense: Spatio-Temporal Fusion for Multi-View 3D Human Pose Estimation with DenseWarper** (arXiv 2026)  
  Link: <https://arxiv.org/abs/2605.14525>  
  *Key idea:* Sparse interleaved input plus DenseWarper module that warps dense temporal features across views.  
  *Relevance:* Most directly relevant to v26. Suggests a lightweight temporal warper before triangulation instead of a heavy full ST transformer.

### Structural / State-Space Fusion

- **MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation** (CVPR 2025)  
  Link: <https://arxiv.org/abs/2509.00649>  
  *Key idea:* Replace vanilla cross-view attention with a Projective State Space (PSS) block and Grid Token-guided Bidirectional Scan (GTBS) to preserve spatial structure and generalise across camera arrangements.  
  *Relevance:* Strongest temporal/structural alternative to our transformer fusion. Supports variable-view inference and could inform a v26 state-space cross-view aggregator.

### Generative & Differentiable Rendering

- **DisPOSE: Projected Polystochastic Diffusion for Self-Supervised Multi-View 3D Human Pose Estimation** (arXiv 2026)  
  Link: <https://arxiv.org/abs/2606.07419>  
  *Key idea:* Diffusion over projected person assignments; hypergraph-convolutional decoder regresses 3D skeletons without dense 3D labels.  
  *Relevance:* Generative assignment prior could extend the pipeline to multi-person scenes and could augment the v20 diffusion refiner.

### Calibration & Camera Robustness

- **RUMPL: Ray-Based Transformers for Universal Multi-View 2D to 3D Human Pose Lifting** (arXiv 2025)  
  Link: <https://arxiv.org/abs/2512.15488>  
  *Key idea:* Represent 2D keypoints as 3D rays; a view-fusion transformer aggregates along rays and is camera- and view-count agnostic.  
  *Relevance:* Validates our v17 ray/camera embeddings and v25 ray-token design. The universal-lifting goal aligns with variable-view training.

- **Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation** (arXiv 2025)  
  Link: <https://arxiv.org/abs/2503.11652>  
  *Key idea:* Exploits additional body-worn/rear cameras to overcome self-occlusion in egocentric HMD setups.  
  *Relevance:* Highlights camera-layout diversity; our variable-view training can simulate non-frontal camera setups.

- **RPGD: RANSAC-P3P Gradient Descent for Extrinsic Calibration in 3D Human Pose Estimation** (arXiv 2026)  
  Link: <https://arxiv.org/abs/2602.13901>  
  *Key idea:* Hybrid RANSAC + P3P gradient descent to recover extrinsics from pose estimates.  
  *Relevance:* Reinforces value of camera-parameter conditioning and principal-point correction in v3/v5 architecture.

### Other Notable Methods

- **COMPOSE: Hypergraph Cover Optimization for Multi-view 3D Human Pose Estimation** (arXiv 2026)  
  Link: <https://arxiv.org/abs/2601.09698>  
  *Key idea:* Training-free hypergraph exact-cover optimization over person hypotheses; solves correspondence and pose jointly via ILP / Belief Propagation.  
  *Relevance:* Informs our v21/v24 neural BA and robust triangulation fallback strategies. Potential multi-person extension.

- **SkelSplat: Robust Multi-view 3D Human Pose Estimation with Differentiable Gaussian Rendering** (WACV 2026)  
  Link: <https://arxiv.org/abs/2511.08294>  
  *Key idea:* Model skeleton as 3D Gaussians and optimize via differentiable rendering; no 3D ground truth required.  
  *Relevance:* Differentiable rendering prior could regularize v21 neural BA or v22/v23 SMPL / KAP branches.

## 3. Concrete v26 Design Recommendations

### 3.1 Keep the deformable spatio-temporal attention lightweight

The v26 proposal (``docs/proposals/v26_temporal_fusion.md``) defines a sparse ``DeformableSpatioTemporalAttention`` block. The 2025-2026 literature reinforces:

- Use **local temporal offsets** (e.g. `[-1, 0, +1]`) first; DenseWarper shows dense warping can be added later without a full ST transformer.
- Add a **motion cost** based on reprojection residual variance across views, as proposed; DeProPose-style relative projection error can be used as an additional gating signal.
- Ensure **identity-at-init** so v26 can warm-start from v18/v23 checkpoints without regressing the 20.24 mm baseline.

### 3.2 Plan an MV-SSM fallback experiment

MV-SSM (CVPR 2025) is the strongest signal that attention may not be the only cross-view aggregator. A minimal follow-up would be:

- Add a toggle ``use_state_space_cross_view=True`` in ``omniview_fusion_v5.py``.
- Implement a 1-D Mamba scan over views (or a simplified S4 block) as a drop-in replacement for one geometry-attention layer.
- Run a smoke test on H36M with 2-4 views and compare to the v25 transformer attention.

### 3.3 Integrate adaptive deficiency weighting

DeProPose (2025) and UPose3D (2024) both argue for uncertainty-aware fusion. v26 should:

- Compute per-frame, per-view, per-joint deficiency scores from the temporal context.
- Use these scores to gate the contribution of each temporal offset before aggregation.
- This naturally extends the v25 confidence/reprojection weighting into the temporal domain.

### 3.4 Reserve generative priors for v27+

SkelSplat and DisPOSE are promising but heavier. They should be tracked as future directions, not as part of the v26 minimal change.

## 4. Open Questions

1. Does the DenseWarper-style dense temporal warper outperform the sparse deformable sampler on WebBridge variable-view clips?
2. Can an SSM block match or exceed v25 geometry attention on H36M while using less memory?
3. How should deficiency weighting interact with the v25 learned depth-proposal head?

## 5. Sources

This digest was generated by ``scripts/summarize_2025_2026_multiview_pose.py`` from the project's existing literature reviews:

- `docs/literature_review_multiview_pose.md`
- `docs/swarm_iter23/related_work_survey.md`