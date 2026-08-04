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

## Quick start

```bash
cd /path/to/repo
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -v
```

## Current status

- ✅ Iteration 1 closed (#1): 20 parallel research notes, v1 design doc, minimal DLT skeleton.
  - Issue: https://github.com/adjcjh777/motionflow-multiview-kimiswarm/issues/1
- 🔄 Iteration 2 open (#2): end-to-end baseline on Shelf/Campus.
  - Issue: https://github.com/adjcjh777/motionflow-multiview-kimiswarm/issues/2

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
