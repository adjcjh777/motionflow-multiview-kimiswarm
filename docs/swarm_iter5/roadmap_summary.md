# Roadmap Summary (swarm_iter5)

This folder contains the ICRA/CVPR 2027 roadmap for the MotionFlow multi-view extension.

## What was produced

- `docs/swarm_iter5/roadmap.md`: a ranked list of 11 deliverables by expected impact and effort, plus a clear statement of the single next experiment.

## Key finding

The single most important next step is **not another architecture change**. The v3 model, data (62k H36M frames), and trainer already exist. The highest-impact, lowest-effort move is to:

1. Clean up `experiments/train_ray_attention_v3_h36m.py` (docstring, default path, checkpoint naming).
2. Train `RayAttentionFusionModelV3` on `data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview.npz`.
3. Evaluate against DLT under clean, noisy, occluded, and outlier conditions.

This run establishes whether camera-conditioned embeddings improve upon the strong geometric baseline and unblocks all downstream ablation, robustness, and paper-writing work.
