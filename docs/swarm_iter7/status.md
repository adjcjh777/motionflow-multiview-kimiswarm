# Swarm Iteration 7 Status

**Date:** 2026-08-04

## Current best (verified)

| Dataset | MPJPE (mm) | PA-MPJPE (mm) | AUC | Checkpoint |
|---|---:|---:|---:|---|
| MPI-INF-3DHP (S1→S2/Seq1) | 11.17 | 8.24 | 0.9256 | `outputs/ray_attention_temporal_residual_final5.pth` |
| Human3.6M (S1→S5) | 5.74 | 3.99 | 0.9618 | `outputs/ray_attention_temporal_residual_h36m.pth` |

## Robustness (final5)

| Perturbation | Level | MPJPE (mm) |
|---|---|---:|
| Clean | 0 | 11.17 |
| Gaussian noise | 5 px | 12.96 |
| Gaussian noise | 20 px | 28.00 |
| Joint occlusion | 50% | 11.18 |
| 2D outliers | 20% | 15.13 |

## Ongoing experiments

| Task | Status | Output |
|---|---|---|
| Full 5-epoch uncertainty-aware residual | running (solo) | `outputs/ray_attention_temporal_residual_uncertainty_full5.pth` |
| Longer temporal window (clip_len=27) smoke | queued after uncertainty | `outputs/ray_attention_temporal_residual_clip27_smoke.pth` |
| Cross-view scaled full run | planned | `docs/swarm_iter7/plan_crossview_residual_scaled_mpi_run.md` |
| Iterative residual refinement | planned | `docs/swarm_iter7/plan_iterative_residual_unrolled_damping.md` |
| Real-world GVHMR demo | planned | `docs/swarm_iter7/plan_real-world_GVHMR_demo.md` |

## Artifacts

- Paper draft: `docs/paper_draft_icra_cvpr_2027.md`
- Architecture figure: `docs/figures/architecture.png`
- Verified results JSON: `docs/swarm_iter7/verified_results.json`
- GitHub issue/PR drafts: `docs/swarm_iter6/github_issue_draft.md`, `docs/swarm_iter6/github_pr_draft.md`

## Blockers

- GitHub issue/PR automation requires `gh auth login` (Personal Access Token or browser auth).
