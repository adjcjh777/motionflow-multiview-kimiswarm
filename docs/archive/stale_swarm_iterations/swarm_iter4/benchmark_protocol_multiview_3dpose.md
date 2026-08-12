# Benchmark Protocol for Multi-View 3D Pose (Shelf/Campus Splits, Cross-Dataset Generalization)

## 1. Brief Survey

Calibrated multi-view 3D human pose benchmarks in the literature are dominated by **Shelf** [1] and **Campus** [2] from the VoxelPose distribution, plus **Human3.6M** [3] and **CMU Panoptic** [4] for 3D-ground-truth (3D-GT) evaluation. Shelf and Campus are attractive because they provide multi-camera calibration and per-frame 3D skeleton annotations, but they are small: Shelf_Seq1 has only ~3 000 frames and Campus_Seq1 ~1 800 frames. Human3.6M offers a canonical train/test split (S1,5,6,7,8 train; S9,11 test) but is captured by four synchronized cameras with limited viewpoint diversity. Panoptic provides denser camera arrays but stricter licensing and heavier preprocessing.

The current MotionFlow-MV stack already consumes Shelf/Campus via `motionflow_mv/data/shelf_loader.py` and trains `ray_attention` on real GT with `experiments/train_ray_attention_real.py`. Design v3 reports that geometric baselines (DLT, robust triangulation) generalize across Shelf and Campus (≈1.5 px reprojection on Campus when trained on Shelf), while the learned `attention` model fails zero-shot (318 px) and `residual_refiner` only transfers moderately (8.26 px). This confirms that a rigorous **split-and-generalize benchmark** is needed before ICRA/CVPR 2027, not merely a single-dataset reprojection leaderboard.

## 2. Concrete Recommendations

### R1. Adopt a fixed temporal split, not a random one
`train_ray_attention_real.py` currently shuffles all frames randomly and holds out 20%. For multi-view pose, temporal independence is weak; random splitting leaks motion dynamics and overstates performance. Instead, split each sequence by **contiguous temporal blocks** (e.g., first 70% train, next 10% validation, final 20% test). Report per-sequence results and a pooled mean.

### R2. Standardize cross-dataset generalization as a primary metric
Define two cross-dataset tasks:
- **Shelf → Campus**: train on Shelf_Seq1, evaluate MPJPE / reprojection on Campus_Seq1.
- **Synthetic → Real**: train on the existing synthetic generator, evaluate on Shelf/Campus without fine-tuning.
Report **MPJPE (mm)**, **Procrustes-aligned MPJPE (PA-MPJPE)**, and **PCK@150 mm** for both. This directly tests whether the ray-aware attention weights are geometry-aware or merely dataset-specific.

### R3. Add a deterministic DLT ceiling and per-joint diagnostic breakdown
Every learned fusion result must be reported alongside the confidence-weighted DLT baseline computed from the same 2D inputs. In addition, break errors by body part (head/torso/arms/legs) and by view count (2-view, 3-view, 4-view cases). The ray-aware model justifies its existence only if it outperforms DLT in occlusion-heavy subsets while matching DLT on clean 4-view frames.

### R4. Use metric units and camera-consistent scaling everywhere
The codebase converts Shelf/Campus from millimeters to meters (`joints_3d / 100.0`) and optionally scales intrinsics (`args.input_scale`). Document this explicitly in the benchmark: all 3D metrics are reported in **millimeters**, and intrinsics/2D coordinates must share the same pixel unit. Store the canonical `length_unit` in the `HumanMotionIR` metadata to prevent cross-dataset unit regressions.

### R5. Reserve a held-out “stress-test” protocol
Beyond standard splits, create an occlusion stress test by artificially dropping views according to visibility heuristics (zero confidence for 1–2 views, 2D outlier injection). Report MPJPE degradation curves. This matches the synthetic ablation in `eval_ray_attention_robustness.py` and gives a clear signal on whether learned weights are robust.

## 3. Potential Risks

- **Dataset scale**: Shelf and Campus together are too small to train or validate a deep model conclusively. Recommend treating them as a **development and ablation benchmark**, while Human3.6M/Panoptic remains the primary 3D-GT benchmark.
- **Unit/calibration mismatch across datasets**: The ray-aware model relies on correct intrinsics and world-scale extrinsics. A silent mm→m error or principal-point offset can dominate MPJPE. Automated calibration validation must run before any numbers are reported.
- **Overfitting to synthetic data**: The synthetic generator produces perfect ground-truth poses and controlled noise. Real datasets contain motion blur, detector failure, and body-shape variation. Synthetic pre-training should be treated as initialization, not a final result.
- **Learned fusion not beating DLT**: Current evidence shows `ray_attention` is strong synthetically but untested on real data. If it does not beat DLT on Shelf/Campus, the paper’s contribution must be reframed as a *modular, uncertainty-aware fusion study* rather than raw accuracy improvement.

## 4. Fit into the Paper Plan

This benchmark protocol directly supports the paper’s two headline claims: (1) a **modular multi-view fusion plugin** under a common `HumanMotionIR`, and (2) **geometry-aware learned fusion that generalizes across calibrations**. The protocol is the evidence layer: it will produce the ICRA/CVPR 2027 tables comparing DLT, robust triangulation, residual refiner, and ray-aware attention on Shelf/Campus splits and cross-dataset transfer. Following this protocol also forces the implementation of `motionflow_mv/eval/metrics.py` and `experiments/compare_fusion_h36m.py`, which are required deliverables in design v2.

## References

1. Belagiannis et al., *3D Pictorial Structures for Multiple Human Pose Estimation*, CVPR 2014.
2. Belagiannis et al., *Multiple Human Pose Estimation with Temporally Consistent 3D Pictorial Structures*, ICPR 2014.
3. Ionescu et al., *Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing*, T-PAMI 2014.
4. Joo et al., *Panoptic Studio: A Massively Multiview System for Social Motion Capture*, ICCV 2015.
