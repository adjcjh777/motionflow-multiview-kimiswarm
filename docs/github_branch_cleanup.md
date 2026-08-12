# GitHub Branch Audit & Cleanup Plan

**Audit date:** 2026-08-12
**Audited by:** MotionFlow Agent
**Repository:** `D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`

## 1. Scope

Document the current local and remote branch state, flag any stale branches for deletion, and provide safe deletion commands and rationale. This is a lightweight, low-risk cleanup; no running A800 training or evaluation jobs are touched.

## 2. Commands Used

```bash
# Local and remote branches
git branch -a

# Detailed branch metadata
git for-each-ref --format='%(refname:short) %(committerdate:short) %(committername)' refs/heads refs/remotes

# Verbose tracking info
git branch --no-color -vv

# Worktrees (to avoid deleting a branch checked out elsewhere)
git worktree list --porcelain
```

## 3. Current Branch Inventory

### Local branches

| Branch | Last Commit Date | Author | Tracking | Notes |
|--------|------------------|--------|----------|-------|
| `main` | 2026-08-12 | MotionFlow Agent | `origin/main` | Default branch, currently checked out |

### Remote branches

| Branch | Last Commit Date | Notes |
|--------|------------------|-------|
| `origin/HEAD` | 2026-08-12 | Points to `origin/main` |
| `origin/main` | 2026-08-12 | Only remote branch |

### Worktrees

| Path | HEAD | Branch |
|------|------|--------|
| `D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm` | `7f33145` | `refs/heads/main` |

## 4. Stale-Branch Criteria

A branch is considered stale when it meets **any** of the following:

1. **Merged into `main`** — the branch’s commits are already reachable from `main`.
2. **Abandoned / no longer relevant** — e.g., experiment branches from a completed paper deadline.
3. **Older than 90 days** without recent commits and not protected.
4. **Superseded** by another branch with the same purpose.
5. **Orphaned feature branch** — no open issue/PR, no recent activity.

Protected branches (e.g., `main`, release branches) are never flagged for deletion.

## 5. Stale Branches Identified

**None.**

The repository currently contains only `main` (local) and `origin/main` (remote). There are no feature, experiment, or hotfix branches that meet the stale criteria above.

## 6. Deletion Commands and Rationale

Because no stale branches exist, no deletions are required today. The commands below are kept as templates for future cleanup rounds, and each is annotated with the rationale and safety check it performs.

### Local branch deletion (dry-run then real)

```bash
# List local branches already merged into main (safe candidates)
git branch --merged main

# Delete a single local branch that has been merged
git branch -d <branch-name>

# Force-delete a local branch that has NOT been merged (use with caution)
git branch -D <branch-name>
```

**Rationale:** `git branch -d` refuses to delete a branch that has unmerged commits, preventing accidental loss of work. Use `-D` only after confirming the branch is truly obsolete.

### Remote branch deletion

```bash
# Delete a remote branch on GitHub
git push origin --delete <branch-name>

# Prune stale remote-tracking branches that no longer exist on the remote
git fetch --prune
```

**Rationale:** Removing the remote branch and then pruning ensures local `origin/*` references stay in sync and do not clutter `git branch -a` output.

### Bulk cleanup helpers

```bash
# Show all remote-tracking branches whose remote partner no longer exists
git branch -r --merged main | grep -vE 'HEAD|main'

# Delete every local branch already merged into main (review before running)
# git branch --merged main | grep -v '^\*' | grep -v 'main' | xargs -r git branch -d
```

**Rationale:** These are one-liners for periodic housekeeping; always inspect the list before deleting en masse.

## 7. Safety Checklist

Before deleting any branch:

- [ ] Verify the branch is not the current checkout (`git branch --show-current`).
- [ ] Confirm the branch is not checked out in another worktree (`git worktree list`).
- [ ] Check that the branch has no unique, valuable commits (`git log main..<branch-name>`).
- [ ] Ensure the branch is not protected on the remote (GitHub branch protection rules).
- [ ] Prefer `git branch -d` over `-D` unless the branch is intentionally being discarded.

## 8. Action Log

| Action | Date | Result |
|--------|------|--------|
| Audit local and remote branches | 2026-08-12 | Only `main` / `origin/main` exist |
| Identify stale branches | 2026-08-12 | None found |
| Delete branches | 2026-08-12 | N/A — nothing to delete |

## 9. Next Steps

- Re-run this audit after large paper pushes or experiment campaigns.
- Re-visit this file before the CVPR 2027 submission deadline to clean up any short-lived ablation or SOTA-baseline branches that may accumulate.
- Keep branch names descriptive and short-lived; prefer `feature/`, `ablation/`, `fix/`, or `exp/` prefixes to make future stale-branch identification easier.
