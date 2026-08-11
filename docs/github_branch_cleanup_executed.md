# GitHub Branch Cleanup — Executed

- **Execution date:** 2026-08-11
- **Repository:** `D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm`
- **Executed from branch:** `main`
- **Plan:** [`docs/github_branch_cleanup.md`](github_branch_cleanup.md)

## Summary

Local branch count was reduced from **96** to **15**.

| Action | Count |
|--------|-------|
| Local-only branches archived to `origin` | 12 |
| Branches deleted locally | 74 |
| Branches retained intentionally | 6 |
| Branches left due to active worktrees | 9 |

## Branches archived to origin before deletion

These local-only branches were pushed to `origin` so the commits remain reachable, then deleted locally.

| Branch | Tip on origin |
|--------|---------------|
| `feat/iter-next-audit-webbridge-mpi-inf-3dhp-data-availability-wt` | `baa26b3` |
| `feat/set-transformer-crossview` | `4164fd8` |
| `feat/fast-epipolar-bias-v2-pp` | `4fe4d05` |
| `feat/temporal-skeleton-consistency-loss` | `c0c548a` |
| `feat/multiscale-temporal-residual` | `273b9b5` |
| `run/visibility-uncertainty-v1` | `935a6a7` |
| `feature/webbridge-mixed-17joint-v3` | `9418edc` |
| `clean-data-aug` | `3d75dba` |
| `attention-entropy-interpretability` | `d9922bc` |
| `feature/graph-joint-relation-full-run` | `629e111` |
| `feature/splatv2-view-dependent-covariance` | `b9a65c2` |
| `feature/webbridge-mixed-17joint` | `7e7cb95` |

## Exact duplicate tip groups resolved

For each group, the canonical branch was kept (or already existed on origin) and the duplicate local names were deleted.

| Canonical | Tip | Duplicates deleted |
|-----------|-----|--------------------|
| `feat/iter17-extended-camera-perturbation-curriculum` | `fd6bec5` | `feat/iter17-semi-supervised-pseudo-labeling` |
| `feat/iter-next-prototype-deeper-st-attention` | `5e3432e` | `wt-proto-14460` |
| `feat/iter-next-audit-webbridge-mpi-inf-3dhp-data-availability` | `ef1f6ac` | `feat/iter-next-extend-camera-perturbation-ranges-and-intrinsics-curriculum`, `feat/iter-next-roadmap` |
| `clean-data-aug` | `3d75dba` | `feature/splatv2-view-dependent-covariance-clean2` |
| `feature/splatv2-view-dependent-covariance-final` | `988a63e` | `feature/splatv2-view-dependent-covariance-clean` |

## Stale remote-backed branches deleted locally

All other branches recommended for deletion in `docs/github_branch_cleanup.md` were removed from the local checkout. They remain available on `origin` and can be restored with `git checkout -b <branch> origin/<branch>` if needed.

A representative sample of deleted branches:

- `swarm/*` prototype branches (v17, v18, v19, v20, v21, v22, v26, v27, v28, outlier view, SMPL prior, webbridge, etc.)
- `feat/iter17-*` ablation branches (confidence-aware dropout, splatv2 covariance, visibility uncertainty, epipolar bias, EMA checkpoints, etc.)
- `feat/iter-next-*` experiment branches (trainer cosine warmup, synthetic occlusion, temporal consistency, synchronized augmentation, etc.)
- `fix/eval-v4-geometry-toggles`
- `my-attention-entropy`, `realtime-kd-student-iter16`, `ssl-view-contrast`, `domain_adaptation_shelf_campus_v2`, etc.

See the full deletion log: `tmp/github_branch_cleanup.log`.

## Remaining local branches (15)

### Intentionally kept (6)

| Branch | Reason |
|--------|--------|
| `main` | Default branch |
| `v33-uncertainty-aware-triangulation` | Active release branch (1 ahead; rebase/merge before CVPR) |
| `feat/v29-self-evolving-hierarchical-multiview` | Active feature branch (1 ahead; rebase/merge before CVPR) |
| `fix/h36m-corrected-track` | Local-only data-foundation fix aligned with current true-GT work |
| `feat/unified-results-csv` | Local-only eval utility (results.csv logger) |
| `feat/bayesian-tri-v2-batched-lstsq` | Local-only Bayesian tri v2 eval support |

### Blocked by active/prunable worktrees (9)

These branches could not be deleted because they are currently checked out in git worktrees. They are otherwise stale and could be removed once the worktrees are cleaned up.

| Branch | Worktree path |
|--------|---------------|
| `swarm/v18_deformable_attention_baseline` | `.worktrees/v18_deformable_attention_baseline` |
| `swarm/v20_diffusion_refiner_prototype` | `D:/WSL_workspace/about_eassys/motionflow-v20-tmp` |
| `swarm/v22_kap_integration` | `.worktrees/v22_kap_integration` |
| `swarm/adaptive_view_selector_tuning` | `.worktrees/adaptive_view_selector_tuning` |
| `swarm/webbridge_data_expansion` | `.worktrees/webbridge_data_expansion` |
| `feat/iter17-attention-entropy-regularization` | `.worktrees/iter17-attention-entropy-regularization` |
| `feat/iter17-cross-view-graph-attention` | `D:/WSL_workspace/about_eassys/motionflow-iter17-cross-view-graph-attention` |
| `feat/iter-next-audit-webbridge-mpi-inf-3dhp-data-availability-wt` | `D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm-audit-wt` (prunable) |
| `feat/iter-next-ensemble-inference-multi-checkpoint` | `D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm-ensemble-wt` (prunable) |

## Verification commands

```bash
# Verify local branch count
git for-each-ref --format='%(refname:short)' refs/heads | wc -l

# Verify remaining branches
git for-each-ref --format='%(refname:short) %(objectname:short)' refs/heads | sort

# Verify archived branches on origin
git ls-remote --heads origin | grep -E "(set-transformer|fast-epipolar|temporal-skeleton|multiscale-temporal|visibility-uncertainty|webbridge-mixed-17joint|clean-data-aug|attention-entropy-interpretability|graph-joint-relation-full-run|splatv2-view-dependent-covariance|iter-next-audit-webbridge-mpi-inf-3dhp-data-availability-wt)"

# Inspect worktrees
git worktree list --porcelain
```

## Notes / blockers

1. **Worktrees blocked deletion of 9 stale branches.** The script used `git branch -D`, which refuses to delete a branch that is checked out in a worktree. Two of those worktrees are marked `prunable` (their gitdir files point to non-existent locations), so `git worktree prune` could allow their removal. The other 7 worktrees still exist on disk and should not be removed without explicit confirmation.
2. **No GPU jobs were started or stopped** during this cleanup; it is purely a repository-maintenance task.
3. **No git mutations were performed on A800-D**; all operations stayed in the local repository and its configured `origin` remote.
4. The repository had uncommitted changes at the time of cleanup (modified docs, new configs, etc.), which were left untouched.

## Full execution log

`tmp/github_branch_cleanup.log`
