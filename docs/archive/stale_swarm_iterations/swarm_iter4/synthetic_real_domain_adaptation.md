# Synthetic-to-Real Transfer and Domain Adaptation Strategies

**Topic:** `synthetic_real_domain_adaptation`  
**Scope:** Multi-view 3D human pose fusion, synthetic pre-training, real-world fine-tuning.  
**Target venues:** CVPR / ICRA 2027.  
**Date:** 2026-08-04.

---

## 1. Problem statement

`RayAttentionFusionModel` is metric-stable on synthetic data, yet the project lacks a clear path from synthetic pre-training to real Shelf/Campus and Human3.6M. The core question is: **how do we train a learned multi-view fusion model on cheap synthetic SMPL data and adapt it to real datasets without overfitting or suffering a large domain gap?**

`docs/design_v3.md` shows the ray-aware head reaches millimeter-level accuracy on synthetic rigs, while older attention models trained on Shelf still lag DLT. The architecture is sound, but the training recipe—domain-matched synthesis, adaptation losses, and cross-dataset validation—needs to be formalized.

## 2. Brief survey

*   **AMASS / SMPL synthetic data** (Mahmood et al., ICCV 2019) provides realistic poses, closing the motion-distribution gap.
*   **Domain randomization** (Tobin et al., IROS 2017) randomizes cameras, noise, occlusion, and backgrounds so real data falls inside the training distribution.
*   **Domain-adversarial training** (Ganin et al., JMLR 2016) aligns source and target features via a gradient-reversal layer.
*   **Self-supervised geometric losses** (Kocabas et al., CVPR 2019) use multi-view reprojection, epipolar consistency, bone-length, and temporal smoothness when 3D GT is absent.
*   **Progressive transfer** pre-trains on synthetic data, fine-tunes on the target domain, then refines on a small 3D-GT dataset.

## 3. Current codebase mapping

*   `experiments/generate_synthetic_multiview_dataset.py` already produces randomized SMPL rigs, but samples Brownian-motion poses instead of AMASS motions.
*   `motionflow_mv/fusion/ray_attention_model.py` accepts per-sample `(K, R, t)`, making domain randomization easy.
*   `experiments/train_ray_attention_real.py` and `experiments/train_ray_attention_synthetic.py` are disjoint; there is no transfer script that loads a synthetic checkpoint and fine-tunes on real data.
*   `docs/design_v3.md` flags cross-dataset generalization as a weakness; the legacy `attention` model fails to transfer from Shelf to Campus.

## 4. Concrete recommendations

### 4.1 Upgrade the synthetic generator with domain-matched distributions

Replace random Brownian motion in `generate_synthetic_multiview_dataset.py` with **AMASS pose sampling**, and match real camera statistics: sample focal lengths, principal points, and baselines from Shelf/Campus/Human3.6M; use detector-realistic noise calibrated to HRNET / Mask R-CNN errors; and add occlusion/outliers at observed real-data rates.

### 4.2 Add a domain-adversarial head during real fine-tuning

Attach a small domain classifier to the per-joint attention features with gradient reversal. Because the DLT triangulation layer is purely geometric, only the view-weighting head needs to learn domain-invariant embeddings. This is a minimal, architecture-preserving change.

### 4.3 Implement progressive transfer with self-supervised real losses

Create `experiments/train_ray_attention_transfer.py`: (1) pre-train on large synthetic AMASS data with 3D MSE; (2) fine-tune on Shelf/Campus with 3D MSE (if available) + multi-view reprojection + bone-length + temporal smoothness; (3) optionally refine on Human3.6M with real 3D GT.

### 4.4 Standardize scale/camera invariance in the model

Normalize inputs before entering `RayAttentionFusionModel`: scale 2D points and intrinsics by image size, and normalize camera centers by rig diameter. Finalize the `input_scale` / `output_scale` plugin contract from `docs/design_v3.md`.

### 4.5 Add cross-dataset validation as a first-class metric

Do not report only Shelf numbers. Establish `train_ray_attention_real.py --train_dataset shelf --val_dataset campus`, report zero-shot MPJPE / reprojection on Campus, and maintain a held-out synthetic test set with unseen camera rigs. Cross-dataset validation confirms whether domain adaptation works.

## 5. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **AMASS licensing** | Limits public release of synthetic data | Document license; release generation script only |
| **Domain gap persists** | Synthetic noise may not match real detector errors | Use real 2D detection noise samples |
| **Self-supervised loss collapse** | Bone-length / temporal priors can over-smooth | Weight losses carefully; validate against 3D GT |
| **Cross-dataset generalization poor** | Shelf and Campus have different camera layouts | Domain-adversarial training + normalized inputs |
| **No 3D GT for Shelf/Campus locally** | Hard to measure real 3D MPJPE | Parse `annotation_3d.json`; prioritize Human3.6M |

## 6. Fit into the paper plan

Synthetic-to-real transfer is the **training backbone** of the CVPR/ICRA paper. It enables the claim: *a lightweight, geometry-aware attention fusion model can be trained without real 3D labels and still outperform algebraic triangulation on real benchmarks.*

This maps to paper sections:

*   **Method:** Domain-matched synthetic generation + ray-aware attention + domain-adversarial fine-tuning.
*   **Experiments:** Synthetic pre-training ablations, Shelf/Campus fine-tuning, cross-dataset validation, Human3.6M refinement.
*   **Results:** Show that synthetic-only training approaches DLT, and Sim2Real + real fine-tuning beats DLT in 3D accuracy.

## 7. References

1. Mahmood, N., et al. *AMASS: Archive of Motion Capture as Surface Shapes.* ICCV 2019.
2. Tobin, J., et al. *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.* IROS 2017.
3. Ganin, Y., et al. *Domain-Adversarial Training of Neural Networks.* JMLR 2016.
4. Kocabas, M., et al. *Self-Supervised Learning of 3D Human Pose using Multi-view Geometry.* CVPR 2019.
5. Iskakov, K., et al. *Learnable Triangulation of Human Pose.* ICCV 2019.
