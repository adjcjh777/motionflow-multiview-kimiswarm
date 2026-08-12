# Iteration Next Exploration Swarm Plan

Goal: push MotionFlow-MultiView toward ICRA/CVPR 2027 publishable quality on MPI-INF-3DHP, with the near-term target of MPJPE < 8.75 mm on the validation set.

Current anchor experiment:
- `scripts/run_bayesian_tri_v2_large_scale_wsl.sh` training `bayesian_tri_v2_pp` (d=128, residual_hidden=256, n_st_layers=3, 50 epochs, 2000 samples) on RTX 4090.
- Log: `outputs/bayesian_tri_v2_large_scale_mpiinf3dhp.log`
- Eval: `scripts/eval_bayesian_tri_v2_large_scale_wsl.sh`

This swarm explores the next iteration in parallel. Each subtask is independent and should result in a branch from `feat/iter-next-swarm`. Full GPU training is queued behind the current run; for now produce code, CPU smoke tests, and documentation.

## Swarm tasks

1. **Data** – Audit WebBridge MPI-INF-3DHP data availability and create a manifest.
2. **Aug** – Implement synchronized multiview 2D augmentation as a standalone module.
3. **Aug** – Implement synthetic joint occlusion augmentation.
4. **Aug** – Improve random view dropout with per-joint confidence-aware resampling.
5. **Calib** – Extend camera perturbation ranges and intrinsics curriculum.
6. **Trainer** – Add cosine LR schedule, warmup, gradient clipping, and AMP to the trainer.
7. **Trainer** – Add EMA checkpoint save/load support.
8. **Tests** – Add unit tests for Bayesian tri v2 batched DLT.
9. **Arch** – Prototype a deeper ST attention model as a new model class.
10. **Arch** – Prototype a cross-view graph attention fusion module.
11. **Loss** – Add temporal velocity + acceleration consistency loss.
12. **Arch** – Prototype Bayesian tri v3 with learned per-joint precision and refinement.
13. **Inference** – Implement ensemble inference across multiple checkpoints.
14. **Eval** – Extend robustness evaluation matrix across noise, occlusion, and view dropout.
15. **HPO** – Create a hyperparameter search script for large runs.
16. **Docs** – Create ablation study CSV template and plotting script.
17. **Docs** – Draft ICRA/CVPR 2027 paper story in docs/.
18. **Docs** – Update README roadmap with next iteration plan.
19. **Comms** – Update GitHub issue #25 with current status and next steps.
20. **Synthesis** – Synthesize this swarm's outputs into a next-iteration action plan.
