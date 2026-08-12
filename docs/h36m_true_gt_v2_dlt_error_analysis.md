# H36M true-GT v2 DLT baseline error analysis

> Protocol: **S1, S5, S6, S7, S8 train → S9, S11 test**  
> Labels: `data/h36m_true_gt_v2/*_multiview_m.npz` (non-circular true mocap world coordinates)  
> Baseline: confidence-weighted DLT (`outputs/h36m_true_gt_v2/dlt_baseline_h36m_true_gt_v2.json`)  
> Analysis script: `scripts/analyze_h36m_true_gt_v2_dlt_errors.py`  
> Outputs: `outputs/h36m_true_gt_v2/dlt_error_analysis/`

## Summary statistics

| Subject | Frames | Mean MPJPE (mm) | Mean reprojection (px) |
|---:|---:|---:|---:|
| S9  | 83,759 | 29.54 | 3.17 |
| S11 | 57,971 | 21.81 | 2.83 |
| **Combined** | **141,730** | **26.38** | **3.03** |

The combined MPJPE of **26.38 mm** matches the weighted test mean in the baseline JSON and is close to the published **25.67 mm** simple mean, with the small difference coming from frame-count weighting.

## Per-joint MPJPE (mm)

| Joint | S9 | S11 | Combined |
|---|---:|---:|---:|
| pelvis | 10.84 | 10.88 | 10.86 |
| right_hip | 26.36 | 19.13 | 23.40 |
| right_knee | 32.93 | 22.14 | 28.51 |
| right_ankle | 41.54 | 25.42 | 34.94 |
| left_hip | 23.87 | 17.78 | 21.38 |
| left_knee | 33.10 | 26.98 | 30.59 |
| left_ankle | 47.37 | 23.07 | 37.43 |
| spine | 25.70 | 18.29 | 22.67 |
| neck | 29.31 | 18.25 | 24.79 |
| head | 18.13 | 16.50 | 17.46 |
| left_shoulder | 12.31 | 11.51 | 11.98 |
| left_elbow | 35.09 | 20.73 | 29.21 |
| left_wrist | 29.62 | 23.25 | 27.02 |
| right_shoulder | 35.52 | 36.62 | 35.97 |
| right_elbow | 36.25 | 21.79 | 30.33 |
| right_wrist | 30.34 | 23.57 | 27.57 |
| head_top | 33.91 | 34.83 | 34.29 |

![Per-joint MPJPE](outputs/h36m_true_gt_v2/dlt_error_analysis/per_joint_mpjpe.png)

## Per-camera reprojection error (pixels)

| Camera | S9 | S11 | Combined |
|---:|---:|---:|---:|
| cam 0 | 3.29 | 2.98 | 3.16 |
| cam 1 | 2.72 | 2.44 | 2.61 |
| cam 2 | 3.11 | 2.68 | 2.93 |
| cam 3 | 3.57 | 3.20 | 3.42 |

![Per-camera reprojection error](outputs/h36m_true_gt_v2/dlt_error_analysis/per_camera_reproj_error.png)

## Per-frame MPJPE over time

![Per-frame MPJPE](outputs/h36m_true_gt_v2/dlt_error_analysis/per_frame_mpjpe.png)

- The smoothed trends stay low for most frames (~20–40 mm) but show clear bursts where DLT error jumps above 100 mm.
- S9 is systematically noisier than S11.

## Confidence vs. error

| Subject | Frame-level r | Observation-level r |
|---:|---:|---:|
| S9  | **−0.677** | −0.488 |
| S11 | **−0.656** | −0.496 |

![Confidence vs. error](outputs/h36m_true_gt_v2/dlt_error_analysis/confidence_vs_error.png)

- Strong negative correlation at the frame level: lower confidence is a reliable signal of higher 3D error.
- The spread at high confidence shows that confidence alone cannot explain all error; some high-confidence frames are still triangulated poorly (likely due to view ambiguity/occlusion).

## Key findings

1. **Feet/ankles are the hardest joints.** Left ankle (37.43 mm combined), right ankle (34.94 mm), and right shoulder/head_top (≈35 mm) dominate the error budget. Pelvis, head, and shoulders are already triangulated well by DLT (≤18 mm).
2. **Large subject asymmetry on S9.** S9’s left ankle error (47.37 mm) is far higher than S11’s (23.07 mm), suggesting occlusion or motion-blur on that side for subject 9.
3. **Camera 1 is most reliable; camera 3 is worst.** The per-view reprojection error varies by ~0.8 px, with cam 3 being the noisiest and cam 1 the cleanest. A learned model should down-weight cam 3 when it disagrees with the others.
4. **Confidence is a useful uncertainty proxy.** Frame-level r ≈ −0.66 means low-confidence frames are exactly the frames where the baseline struggles most.
5. **Per-frame bursts matter.** Even though the average MPJPE is 26 mm, transient spikes >100 mm indicate frames with severe ambiguity that a learned model could target.

## Implications for model design

- **Focus capacity on distal / ambiguous joints.** The largest headroom over DLT is in ankles, wrists, and elbows. Attention or graph modules should prioritize these joints rather than uniformly improving all joints.
- **Joint-aware confidence weighting.** Because confidence correlates strongly with error, architectures should ingest per-joint confidence explicitly and learn to discount low-confidence observations, not just average them.
- **View reliability modeling.** The consistent 0.8 px spread across views suggests a per-view reliability or outlier-rejection module would help, especially for cam 3. View-conditioning (e.g., count embeddings, view dropout) should treat cameras asymmetrically.
- **Sparse-view fallback should protect feet.** If k<4, the feet are likely to degrade first; a DLT-fallback or model-agnostic sparse-view head should be most conservative for ankle/wrist joints.
- **Temporal smoothing for transient spikes.** Per-frame spikes indicate that a small fraction of frames dominate the error; short temporal windows or consistency checks can suppress these without harming the bulk of easy frames.

## Files produced

- `outputs/h36m_true_gt_v2/dlt_error_analysis/per_joint_mpjpe.png`
- `outputs/h36m_true_gt_v2/dlt_error_analysis/per_camera_reproj_error.png`
- `outputs/h36m_true_gt_v2/dlt_error_analysis/per_frame_mpjpe.png`
- `outputs/h36m_true_gt_v2/dlt_error_analysis/confidence_vs_error.png`
- `outputs/h36m_true_gt_v2/dlt_error_analysis/summary.json`
