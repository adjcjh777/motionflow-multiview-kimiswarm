# GitHub Branch Cleanup Audit

> **Audit date:** 2026-08-12
> **Repository:** `D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm`
> **Auditor:** sub-agent on `main`
> **Scope:** Verify post-cleanup branch state; identify remaining stale remote branches and worktree-blocked local branches.
> **Update:** Two stale remote branches (`feat/v29-self-evolving-hierarchical-multiview`, `v33-uncertainty-aware-triangulation`) were deleted from `origin` and pruned locally.

## 1. Summary

A local-branch cleanup was executed earlier today (see [`docs/github_branch_cleanup.md`](github_branch_cleanup.md) and [`docs/github_branch_cleanup_executed.md`](github_branch_cleanup_executed.md)). This audit confirms the local state and flags what still needs attention.

| Metric | Count |
|--------|-------|
| Local branches | 1 |
| Remote branches (origin) | 2 |
| Remote-only / orphaned branches | 0 |
| Local-only branches | 0 |
| Local branches also on origin | 1 |
| Worktree-blocked stale branches | 0 |

## 2. Local Branches (1)

### 2.1 Intentionally retained (1)

| Branch | Remote? | Notes |
|--------|---------|-------|
| `main` | yes | Default branch; in sync with `origin/main` |

### 2.2 Blocked by active worktrees (0)

No worktree-linked local branches remain. The worktrees listed in the previous audit have already been removed.

## 3. Remote-Only / Orphaned Branches (0 candidates)

No remote-only branches remain on `origin`.

### 3.1 Deleted stale branches

The following remote branches were deleted during this cleanup:

| Branch | Deleted | Reason |
|--------|---------|--------|
| `origin/feat/v29-self-evolving-hierarchical-multiview` | 2026-08-12 | Stale feature branch; superseded by later iterations |
| `origin/v33-uncertainty-aware-triangulation` | 2026-08-12 | Stale release branch; uncertainty work merged/replaced |

Commands used:
```bash
git push origin --delete feat/v29-self-evolving-hierarchical-multiview v33-uncertainty-aware-triangulation
git fetch --prune
```

## 4. Issues Found

1. **Dry-run script miscounted remote-only branches.** `scripts/cleanup_branches_dry_run.sh` failed to strip trailing `\r` from `git branch -r` output on Windows, causing it to flag branches that do have local matches as remote-only. Fixed by adding `${remote_branch//$'\r'/}` and filtering bare `origin` entries.
2. **Dry-run script aborts on empty worktree output.** When no worktree-linked branches exist (other than `main`), the `grep -v` in the worktree summary pipeline returns exit code 1, causing `set -o pipefail` to abort the script. Fixed in `scripts/cleanup_branches_dry_run.sh` by appending `|| true` to the worktree summary pipeline and by computing the worktree count dynamically.
3. **No merged local branches.** `git branch --merged main` returns only `main`, so force-deletion is not needed for any local branch.
4. **Local `main` is in sync with `origin/main`.** The previously unpushed commit has been pushed.

## 5. Recommended Next Steps

1. **No further branch deletion is required.** The two targeted stale remote branches have been removed and all orphaned remote branches are gone.
2. **Run the fixed dry-run script** to verify the cleanup state:
   ```bash
   bash scripts/cleanup_branches_dry_run.sh
   ```
3. **Monitor for new stale branches.** Re-run the audit monthly or after major experiments.

## 6. Verification Commands

```bash
# Local branch count
git for-each-ref --format='%(refname:short)' refs/heads | wc -l

# Remote-only branch count
git for-each-ref --format='%(refname:short)' refs/remotes/origin | wc -l

# Worktrees
git worktree list --porcelain

# Dry-run full report
bash scripts/cleanup_branches_dry_run.sh
```

## 7. Blockers

- No blockers remain.
