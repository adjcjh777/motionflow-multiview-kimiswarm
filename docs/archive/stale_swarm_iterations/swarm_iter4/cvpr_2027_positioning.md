# CVPR 2027 Positioning, Contribution, and Experimental Plan

## 1. Topic survey

Calibrated multi-view human pose estimation has two established families of methods:

- **Geometric triangulation** (DLT, RANSAC, confidence-weighted DLT) is exact, metric-preserving, and hard to beat when 2D keypoints and camera calibrations are clean.  It provides strong inductive bias but no adaptivity to occlusion, noise, or outlier views.
- **Learned fusion** (Learnable Triangulation of Human Pose, MVGFormer, HeatFormer, MV-SSM) replaces fixed triangulation with neural modules.  Recent geometry-aware transformers encode ray directions, camera centers, or projection matrices and either regress 3D coordinates or predict per-view weights fed into a differentiable triangulation layer.

The key trend is to keep the camera geometry inside the network rather than flatten it: ray/camera embeddings plus a differentiable triangulation layer are more stable than direct 3D regression from raw projection parameters.

Our current `motionflow_mv/fusion/ray_attention_model.py` follows exactly this design: it computes camera rays and centers, embeds per-view `(x, y, confidence)` together with ray features, predicts per-view weights via self-attention, and triangulates with a differentiable weighted DLT layer.  Early synthetic validation is extremely strong (MPJPE ≈ 2.1 mm on 4-view 0.5 px noise, and ≈ 4–6 mm under 10% occlusion and 2% outliers), confirming the model preserves metric scale and learns to down-weight corrupted views.  The `train_ray_attention_real.py` script is ready for real Shelf/Campus training once the raw data is available.

## 2. Positioning for CVPR / ICRA 2027

Rather than framing the paper as "a better triangulator than DLT on reprojection error," the strongest story is:

> **A modular, world-grounded multi-view extension of MotionFlow that fuses per-view 2D/3D evidence into a single metric `HumanMotionIR`, with a geometry-aware learned fusion head that is a drop-in replacement for classical DLT.**

This positions us as a **systems / pose-fusion paper** rather than a pure pose-estimation paper, which is a better fit for CVPR/ICRA because it emphasizes:

1. **Modularity**: interchangeable fusion plugins (`dlt`, `robust_triangulation`, `ray_attention`, etc.) under a single IR contract.
2. **Metric world grounding**: explicit camera model and per-plugin input/output scale contract, producing real-world coordinates.
3. **Uncertainty-aware output**: per-view weights, reprojection residuals, and per-joint uncertainty in the IR for downstream robotics.
4. **Empirical rigor**: controlled synthetic-to-real ablations that isolate the effect of ray-aware attention vs. raw projection-matrix regression vs. direct 3D regression.

For ICRA, the robotics angle is the uncertainty-aware, metric-fused `HumanMotionIR` used for retargeting.  For CVPR, the learning angle is the ray-aware attention fusion and the systematic ablation of geometry-aware fusion designs.

## 3. Concrete actionable recommendations

1. **Run the real-data ray-attention trainer immediately.**  Execute `experiments/train_ray_attention_real.py` on Shelf_Seq1 (and Campus_Seq1 if available).  Report both MPJPE and reprojection error, and compare against the existing `dlt`, `attention`, `robust_triangulation`, and `temporal_refiner` checkpoints.  This is the single most important result for the paper.

2. **Add a Human3.6M loader and 3D-GT supervised training.**  Shelf/Campus are too small to train a large fusion model.  Implement `motionflow_mv/data/human36m_loader.py` and train `ray_attention` with a combined loss `L_MPJPE + λ_reproj·L_reproj + λ_bone·L_bone + λ_temp·L_temporal`.  Human3.6M 3D GT is the only clear path to beat DLT in absolute accuracy.

3. **Design a synthetic-to-real ablation study.**  Use `experiments/generate_synthetic_multiview_dataset.py` and `experiments/train_ray_attention_synthetic.py` to quantify, on controlled data:
   - Ray-aware embedding vs. flattened projection-matrix embedding vs. no camera embedding.
   - Direct 3D regression vs. weighted DLT output.
   - DLT pseudo-GT supervision vs. real 3D GT supervision.
   These ablations become the core evidence for why the geometry-aware design works.

4. **Populate `HumanMotionIR.uncertainty` and `provenance`.**  Extend the IR with per-view weights, reprojection residuals, and per-joint standard deviation.  This supports the ICRA robotics angle and differentiates us from pose-only papers.  Keep the fields optional so existing downstream code is unaffected.

5. **Set a hard submission timeline.**  ICRA 2027 deadlines are typically around early-to-mid September 2026; CVPR 2027 deadlines are typically around mid-November 2026.  Target a complete real-data paper draft by August 2026 for ICRA and by October 2026 for CVPR.  If real 3D GT training is not ready in time, fall back to a Shelf/Campus + synthetic + pseudo-label (ScoreHMR) study and explicitly label the limitation.

## 4. Potential risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| No real 3D GT (Human3.6M) available in time | Medium | High | Use ScoreHMR pseudo-3D labels and synthetic pre-training; frame the paper as a modular fusion study. |
| `ray_attention` still ties DLT on small real datasets | Medium | High | Combine 3D loss with bone-length and temporal losses; increase training data with Human3.6M. |
| Calibration mismatch across datasets | Medium | Medium | Normalize rays/camera centers inside the model; validate cross-dataset (e.g., train Shelf, test Campus). |
| Synthetic-to-real domain gap | Medium | Medium | Match noise/occlusion distributions; use AMASS motion and realistic camera rigs. |
| Timeline pressure | High | Medium | Freeze the contribution scope now; reject new fusion variants unless they fit the ablation plan. |

## 5. Fit into the paper plan

The ray-aware attention fusion is the **core technical contribution** of the multi-view extension.  It directly addresses the failure mode identified in earlier iterations: naive attention fusion cannot beat DLT, but a geometry-aware weighted-DLT head can.  The planned experiments will supply:

- **Main results**: `ray_attention` vs. `dlt`, `robust_triangulation`, and prior `attention` variants on real multi-view data.
- **Ablation study**: synthetic controlled experiments proving the value of ray-aware embeddings and weighted DLT.
- **IR / systems contribution**: uncertainty-aware `HumanMotionIR` and the plugin fusion interface.
- **Robotics relevance**: metric world coordinates and per-joint uncertainty for retargeting.

Completing the five recommendations above will provide a complete evidence set for either ICRA or CVPR 2027, depending on which deadline real 3D GT becomes available.
