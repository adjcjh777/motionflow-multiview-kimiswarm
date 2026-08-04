# MotionFlow Multi-View: From Monocular Video to Multi-View Human Motion Fusion

> 从单目视频到多视角人体动作融合的轻量级探索

## Core Idea

Extend the existing MotionFlow pipeline (monocular video → human motion) to accept **multi-view videos** of the same action, fuse per-view 2D/3D pose estimates, and calibrate them into a **common physical space**. The goal is a minimal, reproducible baseline that demonstrates improved robustness/accuracy over single-view inference, and serves as a paper-worthy direction.

## Principles

- **No over-engineering**: start with the simplest fusion model that can validate the idea.
- **Iterative evolution**: design → train → validate → feedback → next round.
- **Open by default**: track everything via GitHub Issues / PRs.

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

Recent additions:
  - `motionflow_mv/data/voxelpose_loader.py`: load VoxelPose Shelf calibration and 2D predictions.
  - `experiments/run_shelf_voxelpose_baseline.py`: real-data DLT pipeline.
  - `experiments/eval_shelf_voxelpose.py`: reprojection-error evaluation.

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
