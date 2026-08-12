#!/usr/bin/env bash
# scripts/cleanup_branches_dry_run.sh
# Dry-run branch cleanup helper for the local WSL repo.
#
# Safety rules:
# - This script NEVER runs `git branch -D`, `git branch -d`, or `git push --delete`.
# - It only prints the commands you would run, so you can review them first.
# - Worktree-linked branches are listed separately and must be removed via
#   `git worktree remove` before the branch can be deleted.
#
# Usage:
#   bash scripts/cleanup_branches_dry_run.sh
#
# After reviewing, run the printed commands manually or pipe them to bash at
# your own risk (e.g. `bash scripts/cleanup_branches_dry_run.sh | bash`).

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAIN_BRANCH="main"
STALE_MONTHS=3      # branches with last commit older than this are "stale"
# Branches to protect from deletion (exact names).
PROTECTED_BRANCHES=("$MAIN_BRANCH")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
heading() {
    echo ""
    echo -e "${BLUE}== $1 ==${NC}"
}

is_protected() {
    local branch="$1"
    for pb in "${PROTECTED_BRANCHES[@]}"; do
        if [[ "$branch" == "$pb" ]]; then
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# 1. Worktree-linked branches (must be removed before branch deletion)
# ---------------------------------------------------------------------------
# Gather worktree-linked branches safely. The `grep -v` can return 1 when all
# lines are filtered (e.g., only the main worktree remains); append `|| true`
# so `set -o pipefail` does not abort the script on empty worktree output.
worktree_branches="$(git worktree list --porcelain | awk '/^branch / {print $2}' | sed 's|refs/heads/||' | grep -v "^${MAIN_BRANCH}$" || true)"
worktree_count="$(echo "$worktree_branches" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"

heading "Worktree-linked branches (${worktree_count})"
if [[ "$worktree_count" -eq 0 ]]; then
    cat <<'EOF'
No worktree-linked branches found.
EOF
else
    cat <<'EOF'
The following branches are currently checked out as git worktrees. You must
remove their worktrees before deleting the branches.

Commands to remove each worktree (does NOT delete the branch yet):

EOF
    git worktree list --porcelain | awk '/^worktree / {path=$2} /^branch / {print path, $2}' | while read -r wt_path branch_ref; do
        branch_name="${branch_ref#refs/heads/}"
        if [[ "$branch_name" == "$MAIN_BRANCH" ]]; then
            continue
        fi
        echo "  # Worktree path: $wt_path"
        echo "  git worktree remove --force \"$wt_path\""
    done
fi

heading "Worktree-linked branch summary"
if [[ "$worktree_count" -eq 0 ]]; then
    echo "  (none)"
else
    echo "The ${worktree_count} worktree-linked branches are:"
    while IFS= read -r branch; do
        [[ -z "$branch" ]] && continue
        echo "  - $branch"
    done <<< "$worktree_branches"
fi

# ---------------------------------------------------------------------------
# 2. List all local and remote branches
# ---------------------------------------------------------------------------
heading "Local branches"
git branch --format='%(refname:short) %(committerdate:short) %(upstream:short) %(upstream:track)' | column -t

heading "Remote branches"
git branch -r --format='%(refname:short) %(committerdate:short) %(upstream:short)' | column -t

# ---------------------------------------------------------------------------
# 3. Identify merged local branches
# ---------------------------------------------------------------------------
heading "Merged local branches (candidates for deletion)"
merged_local=()
while IFS= read -r branch; do
    [[ -z "$branch" ]] && continue
    if is_protected "$branch"; then
        continue
    fi
    merged_local+=("$branch")
done < <(git branch --merged "$MAIN_BRANCH" --format='%(refname:short)')

if [ ${#merged_local[@]} -eq 0 ]; then
    echo "  (none)"
else
    printf '  %s\n' "${merged_local[@]}"
fi

# ---------------------------------------------------------------------------
# 4. Identify stale local branches
# ---------------------------------------------------------------------------
heading "Stale local branches (last commit older than ${STALE_MONTHS} months)"
stale_local=()
while IFS= read -r branch; do
    [[ -z "$branch" ]] && continue
    if is_protected "$branch"; then
        continue
    fi
    # Re-check stale threshold; git --merged already covered some of these.
    stale_local+=("$branch")
done < <(git for-each-ref --sort=-committerdate refs/heads/ --format='%(refname:short)' --merged="$MAIN_BRANCH" 2>/dev/null || true)

# Also include branches whose last commit is older than STALE_MONTHS months.
stale_cutoff="$(date -d "${STALE_MONTHS} months ago" +%s 2>/dev/null || date -v -${STALE_MONTHS}m +%s)"
while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    branch="$(echo "$line" | awk '{print $1}')"
    commit_date_str="$(echo "$line" | awk '{print $2}')"
    commit_ts="$(date -d "$commit_date_str" +%s 2>/dev/null || true)"
    if [[ -z "$commit_ts" ]]; then
        continue
    fi
    if [[ "$commit_ts" -lt "$stale_cutoff" ]] && ! is_protected "$branch"; then
        if [[ ! " ${stale_local[*]} " =~ [[:space:]]${branch}[[:space:]] ]]; then
            stale_local+=("$branch")
        fi
    fi
done < <(git for-each-ref --sort=-committerdate refs/heads/ --format='%(refname:short) %(committerdate:short)')

if [ ${#stale_local[@]} -eq 0 ]; then
    echo "  (none)"
else
    printf '  %s\n' "${stale_local[@]}"
fi

# ---------------------------------------------------------------------------
# 5. Identify remote branches that have no matching local branch (orphaned)
# ---------------------------------------------------------------------------
heading "Remote-only / orphaned branches (candidates for deletion)"
remote_only=()
while IFS= read -r remote_branch; do
    remote_branch="${remote_branch//$'\r'/}"  # strip CR from Windows git output
    [[ -z "$remote_branch" ]] && continue
    # Skip HEAD pointer and bare "origin" entries
    if [[ "$remote_branch" == "origin/HEAD" ]] || [[ "$remote_branch" == "origin" ]]; then
        continue
    fi
    local_name="${remote_branch#origin/}"
    if ! git show-ref --verify --quiet "refs/heads/${local_name}"; then
        remote_only+=("$remote_branch")
    fi
done < <(git branch -r --format='%(refname:short)')

if [ ${#remote_only[@]} -eq 0 ]; then
    echo "  (none)"
else
    printf '  %s\n' "${remote_only[@]}"
fi

# ---------------------------------------------------------------------------
# 6. Print dry-run deletion commands
# ---------------------------------------------------------------------------
heading "Dry-run deletion commands (review before running)"
echo "# The following commands are shown for review ONLY."
echo "# They will NOT be executed by this script."
echo ""

# Local merged branches
if [ ${#merged_local[@]} -gt 0 ]; then
    echo "# Delete local branches already merged into ${MAIN_BRANCH}:"
    for branch in "${merged_local[@]}"; do
        echo "git branch -d ${branch}"
    done
    echo ""
fi

# Local stale branches (force delete)
if [ ${#stale_local[@]} -gt 0 ]; then
    echo "# Force-delete stale local branches (review each one):"
    for branch in "${stale_local[@]}"; do
        echo "git branch -D ${branch}"
    done
    echo ""
fi

# Remote orphaned branches
if [ ${#remote_only[@]} -gt 0 ]; then
    echo "# Delete remote-only branches (runs on origin):"
    for branch in "${remote_only[@]}"; do
        echo "git push origin --delete ${branch#origin/}"
    done
    echo ""
fi

# Prune stale remote-tracking branches
heading "Prune stale remote-tracking refs"
echo "# Run this to remove remote-tracking branches that no longer exist on origin:"
echo "git fetch --prune"
echo ""

heading "Summary"
echo "Total merged local branches:  ${#merged_local[@]}"
echo "Total stale local branches:   ${#stale_local[@]}"
echo "Total remote-only branches:   ${#remote_only[@]}"
