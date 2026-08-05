# Design: GitHub Issue/PR Drafts for the Swarm-Next Research Iteration

## Purpose

This design report documents the rationale and structure for the GitHub issue and PR drafts produced by the 20-agent research swarm for the MotionFlow-MultiView project. The drafts synthesize findings from all 20 subagent tasks into a single tracking issue and a consolidated PR, making it easy for maintainers to open the issue / PR via the `gh` CLI once credentials are available.

## Why an umbrella issue + summary PR?

The swarm generated ~20 parallel design reports and prototypes under `docs/swarm_iter_next/`. To avoid fragmenting the conversation, we package the iteration into:

1. **A GitHub tracking issue** that states the current best result, lists high-potential directions, and links to every subagent deliverable.
2. **A consolidated PR draft** that would commit the swarm deliverables (design reports, prototypes, and any new code) and close the tracking issue.

This mirrors the structure already used in `docs/swarm_iter6/github_issue_draft.md` and `docs/swarm_iter6/github_pr_draft.md`, updated for the next swarm iteration.

## Content decisions

- **Current best model:** `RayAttentionFusionModelTemporalResidual` at **10.46 mm MPJPE** on MPI-INF-3DHP, with the lightweight 66k-param variant at **13.22 mm**.
- **20 subagent deliverables:** grouped by theme (geometry, robustness, scalability, reproducibility, integration).
- **Next steps:** prioritized by expected impact and implementation cost.
- **Blockers:** GitHub CLI auth, A800 runtime access, real-world GVHMR demo data.

## Validation

A small Python validator (`validate_drafts.py`) checks that both drafts contain the required sections and that all referenced subagent deliverables exist on disk. This is a smoke test only; no training is run.
