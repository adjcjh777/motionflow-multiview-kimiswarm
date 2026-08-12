# Commit Bundle — 2026-08-12 v4

> Generated: 2026-08-12T12:06 UTC
> Branch: `main`
> HEAD: `cd92054` — docs: update AGENTS and handoff — AIST++ relaunched after 2D NaN zeroing fix
> Note: Local `main` is at same commit as `origin/main`; this bundle captures uncommitted work.

## Bundle Purpose

Package the large batch of local changes since `cd92054` into a single commit and produce a transportable git bundle. Scope covers the v81–v85 architecture work, SOTA baseline scaffolding, MPI/AIST++ integration, and the updated CVPR 2027 handoff documentation.

## Scope

- v85 random view dropout module and launch/eval scripts.
- v81/v82 temporal-pose-attention modules and eval scripts.
- v83/v84 experimental modules (kept for record, though ablations were dropped).
- SOTA baseline scripts/configs (VoxelPose, MVPose) and adapter code.
- MPI-INF-3DHP and AIST++ integration scripts, DLT baselines, and result docs.
- Updated AGENTS.md, README.md, paper/status/roadmap docs, and true-GT result tables.
- WebBridge loader audit, variable-view inference DLT-fallback, and related tests.

## Excluded

- `archive_cleanup_20260811/` (cleanup artifacts, already ignored)
- `cleanup_safe_20260811.sh` (one-off cleanup helper)
- `docs/commit_bundle_*.md` (bundle drafts)
- `docs/git_status_summary.md` (transient status snapshot)
- `tmp_v25_dlt_fallback.json` (transient output)

## Suggested Commit Message

```text
docs+scripts+feat: v85 view dropout, v81/v82 attention, SOTA baselines, MPI/AIST++ integration

- Add v85 random view dropout module and launch/eval scripts.
- Add v81 temporal-pose-attention and v82 multi-scale temporal-pose-attention
  modules, smoke/medium launchers, and variable-view DLT-fallback evals.
- Add v83/v84 experimental fusion modules for the record.
- Add SOTA baseline scaffolding: VoxelPose and MVPose configs, adapters, and
  run scripts under scripts/sota_baselines/.
- Add MPI-INF-3DHP detected-2D generation, DLT baseline, and result docs.
- Add AIST++ integration: full DLT baseline, medium-run scripts, and H36M
  cross-evaluation scripts.
- Update AGENTS.md, README.md, paper draft, CVPR 2027 status/roadmap docs, and
  true-GT result tables with latest H36M/AIST++/MPI numbers.
- Update motionflow_mv/data/webbridge_loader.py audit behavior and
  motionflow_mv/fusion/variable_view_inference.py DLT-fallback logic.
- Add/extend tests for webbridge loader audit, H36M true-GT pipeline, and
  multiview geometry fusion v25.

Excludes cleanup artifacts, older bundle drafts, and transient output files.
```

## Verification Checklist

- [ ] `git status --short` shows only expected untracked files remaining.
- [ ] No `archive_cleanup_20260811/`, `cleanup_safe_20260811.sh`, or `tmp_v25_dlt_fallback.json` is staged.
- [ ] No `docs/commit_bundle_*.md` or `docs/git_status_summary.md` is staged.
- [ ] `git diff --cached --stat` looks reasonable.
- [ ] `git bundle verify` succeeds.
