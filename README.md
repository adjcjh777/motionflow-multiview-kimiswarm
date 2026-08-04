# MotionFlow Multi-View: From Monocular Video to Multi-View Human Motion Fusion

> 从单目视频到多视角人体动作融合的轻量级探索

## Core Idea

Extend the existing MotionFlow pipeline (monocular video → human motion) to accept **multi-view videos** of the same action, fuse per-view 2D/3D pose estimates, and calibrate them into a **common physical space**. The goal is a minimal, reproducible baseline that demonstrates improved robustness/accuracy over single-view inference, and serves as a paper-worthy direction.

## Principles

- **No over-engineering**: start with the simplest fusion model that can validate the idea.
- **Iterative evolution**: design → train → validate → feedback → next round.
- **Open by default**: track everything via GitHub Issues / PRs.

## Design docs

- `docs/design_v1.md`: initial multi-view fusion design.
- `docs/design_v2.md`: paper direction v2 — why DLT is hard to beat and how to frame the contribution.

## Roadmap (draft)

1. **Baseline**: identify MotionFlow architecture and output format.
2. **Fusion model**: minimal multi-view pose fusion + calibration.
3. **Integration**: plug fusion into MotionFlow inference pipeline.
4. **Experiments**: compare monocular vs. multi-view on a small dataset.
5. **Feedback loop**: open issues for each round.

## Repo structure

```
motionflow-multiview-kimiswarm/
├── docs/                  # design notes and experiment logs
│   ├── design_v1.md       # current architecture decision
│   └── swarm_iter1/       # 20 parallel research notes
├── motionflow_mv/         # core package
│   ├── baseline/          # BasePoseEstimator interface
│   ├── calibration/       # Camera model
│   ├── fusion/            # triangulation / fusion modules
│   ├── eval/              # metrics (MPJPE, PA-MPJPE, PCK)
│   └── pipeline.py        # end-to-end pipeline
├── experiments/           # scripts and configs
├── tests/                 # unit tests
├── requirements.txt
└── README.md
```

## Quick start (WSL + RTX 4090)

```bash
cd /path/to/repo
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # installs torch 2.4.0+cu121
.venv/bin/python scripts/check_gpu.py          # verify 4090 is visible
.venv/bin/python -m pytest tests/ -v
```

If you only need CPU, install the CPU wheel manually:
```bash
.venv/bin/pip install torch==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

## Current status

- ✅ Iteration 1 closed (#1): 20 parallel research notes, v1 design doc, minimal DLT skeleton.
- ✅ Iteration 2 closed (#2): end-to-end DLT on real Shelf/VoxelPose data + AttentionFusion sanity training.
  - Tests: 4/4 passed.
  - Shelf 300–600: 301 frames triangulated; reprojection error mean 199 px (median 115 px) — mismatch indicates cross-view person-ID alignment is the next bottleneck.
- 🔄 Iteration 3 open (#3): cross-view person matching + DLT/Attention comparison.
  - Cross-view person matching implemented; reprojection error dropped to mean 9.88 px / median 5.52 px.
  - GPU-aware training scripts added for synthetic and real Shelf data.
  - ✅ WSL RTX 4090 PyTorch verification passed; pytest 4/4 passed.
  - ✅ Synthetic GPU training converges (val_MPJPE 1.54 after 50 epochs, checkpoint at `outputs/attention_fusion_synthetic.pth`).
  - ✅ Synthetic comparison: AttentionFusion (MPJPE 2.31) outperforms DLT (MPJPE 7.93) on noisy synthetic data, showing learned fusion is more robust to noise.
  - ✅ Real Shelf data training/eval run on RTX 4090 (data found at `tmp/voxelpose-pytorch/data/Shelf`).
  - ✅ 真实 Shelf 重投影误差对比（300-600 帧，5 视图）：
    - DLT: mean 9.88 px / median 5.52 px
    - AttentionFusionV1（MSE on DLT pseudo-GT）: mean 81.70 px / median 62.42 px
    - AttentionFusionV1（combined MSE+MPJPE loss, d=64）: mean 80.42 px / median 58.90 px
    - AttentionFusionV2（camera params input, normalized）: mean 184.29 px / median 173.01 px
    - Fine-tune from synthetic: mean 125.87 px / median 103.80 px
    - Residual/refine + reprojection loss: failed
  - 当前结论：DLT 在该真实数据集上非常强，最小可学习融合模型还无法超越；单纯增大模型、加入相机参数、或微调合成模型均未能追平 DLT，需要更深度架构或 3D 监督。
- ✅ Iteration 5 closed: learned confidence-weighted DLT (`RobustTriangulationModel`).
  - 真实 Shelf 重投影误差对比（300–600 帧，5 视图）：
    - DLT: mean 9.88 px / median 5.52 px
    - RobustTriangulationModel（学习每视角权重 + 可微分 DLT，reprojection loss）：mean 11.64 px / median 5.98 px
  - 结论：可学习权重能接近 DLT（median 仅高 0.46 px），但 mean 因少量离群帧而更高，未能超过 DLT。这说明仅依赖重投影 loss 的自适应权重在缺少 3D 监督时难以超越几何 DLT。
- ✅ Iteration 6 closed: residual refinement on top of DLT (`ResidualRefinerModel`).
  - 真实 Shelf 重投影误差对比（300–600 帧，5 视图）：
    - DLT: mean 9.88 px / median 5.52 px / max 1044.68 px
    - ResidualRefinerModel（DLT + 可学习残差，L1 reprojection loss）：mean 9.90 px / median 5.52 px / max 1038.20 px
  - 结论：残差校正基本复现了 DLT 的性能，mean/median 与 DLT 持平，max 略有下降，但**仍未统计意义上击败 DLT**。简单的单帧残差网络无法克服 DLT 的强几何先验。
- ✅ Iteration 7 closed: temporal refinement over a 5-frame window (`TemporalRefinerModel`).
  - 真实 Shelf 重投影误差对比（300–600 帧，5 视图）：
    - DLT: mean 9.88 px / median 5.52 px / max 1044.68 px
    - TemporalRefinerModel（Bi-GRU over 5 frames, per-joint features）：mean 9.89 px / median 5.49 px / max 1044.45 px
  - 结论：时序模型在 median 上略有提升（5.49 vs 5.52），mean 与 DLT 持平，max 几乎不变。**仍未显著击败 DLT**。说明当前 2D 检测噪声和重投影 loss 下，时序信息只能带来边际收益。
- ✅ Iteration 8 closed: synthetic pre-training + fine-tuning of `TemporalRefinerModel`.
  - 先用 500 组合成 9 帧序列训练 Bi-GRU（3D MSE loss，收敛到 ~0）。
  - 再用 Shelf 真实数据微调（reprojection loss, window=9, lr=1e-4）。
  - 真实 Shelf 重投影误差对比（300–600 帧，5 视图）：
    - DLT: mean 9.94 px / median 5.53 px / max 1044.68 px
    - TemporalRefinerModel (fine-tuned): mean 9.94 px / median 5.53 px / max 1044.67 px
  - 结论：合成预训练也无法让模型在真实 Shelf 上超越 DLT。纯重投影 loss + 合成数据无法提供足够强的 3D 监督来克服 DLT 的几何先验。
  - 下一步（Iteration 9 待探索）：必须使用带 3D GT 的真实数据集（如 Human3.6M 或 Campus GT）或引入显式的人体骨骼 / 运动先验，才能建立可发表论文的优势。

Recent additions:
  - `motionflow_mv/pipeline_utils.py::select_best_person_group`: match the same person across views by minimal reprojection error.
  - `motionflow_mv/data/voxelpose_loader.py`: load VoxelPose Shelf calibration and 2D predictions.
  - `experiments/run_shelf_voxelpose_baseline.py`: real-data DLT pipeline.
  - `experiments/eval_shelf_voxelpose.py`: reprojection-error evaluation for DLT.
  - `experiments/train_attention_fusion.py`: GPU-aware synthetic training.
  - `experiments/train_attention_fusion_shelf.py`: train AttentionFusion on real matched Shelf data (DLT as pseudo-GT).
  - `experiments/eval_attention_fusion_shelf.py`: evaluate trained AttentionFusion on Shelf via reprojection error.
  - `experiments/compare_dlt_attention_synthetic.py`: compare DLT vs trained AttentionFusion on synthetic noisy data.
  - `motionflow_mv/fusion/robust_triangulation.py`: differentiable confidence-weighted DLT triangulation.
  - `experiments/train_robust_triangulation_shelf.py`: train learned per-view weights for triangulation on Shelf.
  - `experiments/eval_robust_triangulation_shelf.py`: evaluate learned triangulation vs DLT.
  - `motionflow_mv/fusion/residual_refiner.py`: per-frame residual refinement on top of DLT output.
  - `experiments/train_residual_refiner_shelf.py`: train the residual refiner with reprojection loss.
  - `experiments/eval_residual_refiner_shelf.py`: evaluate the residual refiner vs DLT.
  - `motionflow_mv/fusion/temporal_refiner.py`: Bi-GRU temporal refiner over a window of frames.
  - `experiments/train_temporal_refiner_shelf.py`: train the temporal refiner.
  - `experiments/eval_temporal_refiner_shelf.py`: evaluate the temporal refiner vs DLT.
  - `experiments/train_temporal_synthetic.py`: pre-train the temporal refiner on synthetic 3D sequences.
  - `experiments/run_multiview_pipeline_shelf.py`: end-to-end multi-view pipeline demo on Shelf.

## Iteration workflow

1. Open / update a GitHub Issue describing the round’s goal.
2. Run a parallel research / implementation swarm.
3. Synthesize findings into a design doc and a PR.
4. Validate on the target hardware (RTX 4090 / A800-D).
5. Collect feedback and start the next round.

## Hardware

- Local WSL: NVIDIA RTX 4090
- Remote: A800-D via SSH

## License

To be determined based on the baseline MotionFlow license.
