# v49 Related-Work Mapping

This table maps each MotionFlow-MultiView component to the literature gap it addresses and the closest comparable prior work.

| Component | What it does | Literature gap | Comparable prior work |
|---|---|---|---|
| v25 Multi-view geometry fusion | Fuses per-view 2D keypoints via projective geometry + DLT | Pure-learning fusion often ignores camera geometry | Metro (Liao et al.), TokenHMR |
| v45 Adaptive geometry fusion | Learns per-view/joint triangulation weights from features | Fixed confidence or uniform triangulation | vP nViews (Iskakov), MVOE |
| v37 Self-critique view reliability | Predicts per-(view,joint) reliability from reprojection residuals | Hard outlier rejection or hand-tuned thresholds | VoxelPose, SHOT |
| v39 Reliability-coupled adaptive refinement | Refinement gated by reliability scores | Refinement heads do not close the reliability loop | PoseRefiner, PoseFormer |
| v43 Adaptive per-node residual | Residual scaled by per-joint uncertainty | Residual connections ignore uncertainty | HRNet-3D, RLEPose |
| v46 Sparse-view generalization | View-dropout + reliability head for variable V | Models assume fixed number of cameras | MVS-Human, MV-SSM |
| v47 Temporal aggregation | Lightweight transformer over (time, joint) tokens | Temporal models are heavy and not causal | MPT, D3DP |
| v48 Domain generalization | Domain-conditional FiLM + gradient reversal + DDWL | Studio-to-in-the-wild transfer is weak | Metadapt, DLOW |
| v49 Real-time streaming | Causal temporal smoother + dynamic view budget | Accurate methods are not latency-aware | MV-SSM, RUMPL |

## Self-evolution feedback loop

```text
Prediction -> Reprojection residual -> Reliability/uncertainty update -> Refined triangulation -> Prediction
```

This loop is the central methodological contribution that ties v37, v39, v43, v45, v46, v47, and v48 together.
