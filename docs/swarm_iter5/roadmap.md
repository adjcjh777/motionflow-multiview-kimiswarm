<!--
Produced by swarm_iter5 roadmap task.
Summary: ICRA/CVPR 2027 deliverable ranking for MotionFlow multi-view ray-attention fusion.
The highest-priority next step is to train ray_attention_v3 on the existing 62k-frame
H36M multi-view dataset and benchmark it against the differentiable weighted DLT baseline
under clean, noisy, occluded, and outlier-corrupted conditions.
-->

# ICRA/CVPR 2027 Roadmap — MotionFlow Multi-View

## 1. Current state

- **Model**: `RayAttentionFusionModelV3` is implemented in `motionflow_mv/fusion/ray_attention_v3_model.py`. It adds camera-conditioned embeddings to the v2 view/joint attention architecture and still triangulates through the differentiable weighted DLT layer.
- **Trainer**: `experiments/train_ray_attention_v3_h36m.py` imports the v3 model but still carries v2-style docstrings, default paths, and checkpoint names.
- **Data**: `data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview.npz` is available locally (62,094 frames, 4 views, 17 joints, K/R/t for a single rig).
- **Baseline**: the differentiable weighted DLT in `motionflow_mv/fusion/ray_attention_model.py`.
- **Status**: v1/v2 have been validated on synthetic, Shelf, Campus, and a small H36M sequence; a full 62k-frame v3 run has not yet been completed.

## 2. Deliverables ranked by impact and effort

| Rank | Deliverable | Impact | Effort | Why it matters |
|---|---|---|---|---|
| 1 | **v3 H36M 62k training + clean/noisy evaluation** | Very High | Low | Script and data already exist; lowest-effort path to a strong real-GT result. |
| 2 | **Robustness evaluation protocol (noise / dropout / outliers)** | High | Low–Medium | Extends #1; generates the controlled curves reviewers expect. |
| 3 | **Reprojection / epipolar auxiliary loss** | High | Medium | Addresses scale/camera-transfer generalization and beats DLT more reliably. |
| 4 | **Bone-length / skeleton consistency loss** | High | Medium | Gives a geometric prior for heavily occluded views. |
| 5 | **Cross-dataset generalization (train Shelf/Campus, test H36M and vice versa)** | High | Medium | Tests whether camera-conditioned embeddings actually transfer across rigs. |
| 6 | **Ablation study: v1 vs v2 vs v3, frozen vs learned DLT weights** | High | Medium | Core paper evidence for the geometry-aware design. |
| 7 | **Multi-view SMPL fitting / parametric body recovery** | Very High | High | Long-term differentiator; turns joint triangulation into a body model. |
| 8 | **ScoreHMR / GVHMR pseudo-GT on real videos** | Medium–High | High | Removes dependency on rare 3D GT; requires detector pipeline. |
| 9 | **Temporal consistency / video smoothing** | Medium | Medium | Useful for robotics downstream; not on the critical path. |
| 10 | **Real-time efficiency / deployment benchmark** | Medium | Medium | Parameter count, FLOPs, latency for ICRA-style deployability claims. |
| 11 | **Paper outline, figures, and benchmark tables** | Medium | Low | Should start once #1–#6 are in place. |

## 3. Single next experiment

**Train `RayAttentionFusionModelV3` on the full 62k-frame H36M multi-view dataset and evaluate it against the differentiable DLT baseline under clean, noisy, occluded, and outlier-corrupted conditions.**

Why this one:

1. **Highest impact-to-effort ratio**: the model, trainer, and dataset are already in place.
2. **Validates the v3 design choice** (camera-conditioned embeddings) on real 3D GT at scale.
3. **Establishes the DLT gap**: does learning beat a strong geometric baseline on clean data? By how much under outliers?
4. **Unblocks everything else**: robustness curves, ablations, and paper figures all need this checkpoint first.

Suggested command (WSL 4090):

```bash
python experiments/train_ray_attention_v3_h36m.py \
    --dataset data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview.npz \
    --epochs 50 --batch_size 32 --d 64 --lr 1e-3
```

Before running, update the script’s docstring, default `--dataset` path, and checkpoint name prefix from `v2` to `v3` so results are not overwritten or mislabeled.

## 4. Dependencies and blockers

- **H36M data**: available locally; no download needed.
- **GPU**: WSL 4090 is sufficient; A800-D is read-only and must not be used for writes or training.
- **Script cleanup**: minor (docstring, defaults, checkpoint naming) in `experiments/train_ray_attention_v3_h36m.py`.
- **Evaluation script**: extend `experiments/eval_ray_attention_robustness_real.py` to accept the v3 H36M checkpoint and report MPJPE plus per-joint/per-view weight statistics.

## 5. Suggested timeline

- **Week 1**: Run the single next experiment (#1) and produce robustness curves (#2).
- **Week 2**: Add reprojection loss (#3) and bone-length loss (#4); re-train and compare.
- **Week 3**: Cross-dataset evaluation (#5) and ablations (#6).
- **Month 2+**: Stretch goals (#7 SMPL fitting, #8 pseudo-GT) only if #1–#6 clearly beat DLT and transfer across calibrations.
