#!/usr/bin/env bash
# Safe cleanup execution per docs/cleanup_plan.md
# Archives (does not delete) tmp/ and outputs/ smoke artifacts.
# Run from repo root.
set -euo pipefail

DATE=$(date +%Y%m%d)
ARCHIVE="archive_cleanup_${DATE}"
REPORT="cleanup_summary_${DATE}.txt"

# ---------------------------------------------------------------------------
# Protected patterns (will NOT be archived)
# ---------------------------------------------------------------------------
PROTECTED=(
    # Active training run (agent-51)
    "outputs/omniview_fusion_v25_h36m_true_gt_medium"
    # Active tail process
    "outputs/v39_rcgr_smoke_local_4090.log"
    # True-GT baselines referenced in docs/results_true_gt_*.md
    "outputs/iskakov_h36m_true_gt"
    "outputs/iskakov_shelf_campus_detected"
    "outputs/omniview_fusion_v80_h36m_true_gt_smoke"
    "outputs/omniview_fusion_v25_shelf_campus_detected_smoke"
    "outputs/omniview_fusion_v57_shelf_campus_detected_smoke"
    "outputs/omniview_fusion_v80_shelf_campus_detected_smoke"
    "outputs/omniview_fusion_v80_shelf_campus_detected_long"
    # Active / keep directories
    "tmp/swarm_iter_next"
    "tmp/reprojgate_smoke.pth"
    "outputs/a800_h36m_reg"
)

is_protected() {
    local path="$1"
    for p in "${PROTECTED[@]}"; do
        if [[ "$path" == *"$p"* ]]; then
            return 0
        fi
    done
    return 1
}

mkdir -p "$ARCHIVE"
exec > >(tee -a "$ARCHIVE/$REPORT") 2>&1

echo "=== Cleanup execution started at $(date -Iseconds) ==="
echo "Archive root: $ARCHIVE"

# ---------------------------------------------------------------------------
# 1. Before metrics
# ---------------------------------------------------------------------------
echo ""
echo "=== Before cleanup ==="
du -sh tmp/ outputs/ 2>/dev/null || true
BEFORE_COUNT=$(find tmp/ outputs/ -type f 2>/dev/null | wc -l)
echo "Total file count (tmp + outputs): $BEFORE_COUNT"

# ---------------------------------------------------------------------------
# 2. Generate manifests
# ---------------------------------------------------------------------------
echo ""
echo "=== Generating manifests ==="
find tmp/ -type f -printf '%s %p\n' | sort -n > "$ARCHIVE/tmp_manifest_${DATE}.txt"
find outputs/ -type f -printf '%s %p\n' | sort -n > "$ARCHIVE/outputs_manifest_${DATE}.txt"
echo "tmp manifest: $ARCHIVE/tmp_manifest_${DATE}.txt"
echo "outputs manifest: $ARCHIVE/outputs_manifest_${DATE}.txt"

# ---------------------------------------------------------------------------
# 3. Identify candidates
# ---------------------------------------------------------------------------
echo ""
echo "=== Identifying smoke candidates ==="
TMP_CAND="$ARCHIVE/tmp_smoke_candidates_${DATE}.txt"
OUT_CAND="$ARCHIVE/outputs_smoke_candidates_${DATE}.txt"

find tmp/ -type f \( -name '*_smoke*.pth' -o -name '*_smoke*.npz' -o -name 'check_*.py' -o -name 'inspect_*.py' \) 2>/dev/null | sort > "$TMP_CAND"
find outputs/ -type f \( -name '*_smoke*.pth' -o -name '*_smoke*.log' -o -name '*_smoke*.json' \) 2>/dev/null | sort > "$OUT_CAND"

# Also explicitly include old benchmark smoke directories under outputs/
find outputs/ -type d \( -name 'benchmark_v25_smoke*' -o -name 'benchmark_v??_smoke*' -o -name '*_smoke_dry' \) 2>/dev/null | sort > "$ARCHIVE/outputs_smoke_dirs_${DATE}.txt"

echo "tmp smoke candidates (raw): $(wc -l < "$TMP_CAND")"
echo "outputs smoke candidates (raw): $(wc -l < "$OUT_CAND")"
echo "outputs smoke dir candidates (raw): $(wc -l < "$ARCHIVE/outputs_smoke_dirs_${DATE}.txt")"

# ---------------------------------------------------------------------------
# 4. Filter protected files
# ---------------------------------------------------------------------------
echo ""
echo "=== Filtering protected files ==="
TMP_SAFE="$ARCHIVE/tmp_smoke_candidates_safe_${DATE}.txt"
OUT_SAFE="$ARCHIVE/outputs_smoke_candidates_safe_${DATE}.txt"
OUT_DIR_SAFE="$ARCHIVE/outputs_smoke_dirs_safe_${DATE}.txt"

: > "$TMP_SAFE"
: > "$OUT_SAFE"
: > "$OUT_DIR_SAFE"

while IFS= read -r f; do
    if ! is_protected "$f"; then
        echo "$f" >> "$TMP_SAFE"
    else
        echo "  [protected] $f"
    fi
done < "$TMP_CAND"

while IFS= read -r f; do
    if ! is_protected "$f"; then
        echo "$f" >> "$OUT_SAFE"
    else
        echo "  [protected] $f"
    fi
done < "$OUT_CAND"

# For directories, protect any that contain an active/protected path
while IFS= read -r d; do
    if ! is_protected "$d"; then
        echo "$d" >> "$OUT_DIR_SAFE"
    else
        echo "  [protected] $d"
    fi
done < "$ARCHIVE/outputs_smoke_dirs_${DATE}.txt"

echo "tmp smoke candidates (safe): $(wc -l < "$TMP_SAFE")"
echo "outputs smoke candidates (safe): $(wc -l < "$OUT_SAFE")"
echo "outputs smoke dir candidates (safe): $(wc -l < "$OUT_DIR_SAFE")"

# ---------------------------------------------------------------------------
# 5. Archive candidates (mv, preserving relative path)
# ---------------------------------------------------------------------------
echo ""
echo "=== Archiving candidates ==="

archive_file() {
    local src="$1"
    local dst="$ARCHIVE/$src"
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
}

archive_dir() {
    local src="$1"
    local dst="$ARCHIVE/$src"
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
}

ARCHIVED_TMP=0
while IFS= read -r f; do
    if [ -e "$f" ]; then
        archive_file "$f"
        ((ARCHIVED_TMP++)) || true
    fi
done < "$TMP_SAFE"

ARCHIVED_OUT=0
while IFS= read -r f; do
    if [ -e "$f" ]; then
        archive_file "$f"
        ((ARCHIVED_OUT++)) || true
    fi
done < "$OUT_SAFE"

ARCHIVED_DIRS=0
while IFS= read -r d; do
    if [ -e "$d" ]; then
        archive_dir "$d"
        ((ARCHIVED_DIRS++)) || true
    fi
done < "$OUT_DIR_SAFE"

echo "Archived tmp files: $ARCHIVED_TMP"
echo "Archived outputs files: $ARCHIVED_OUT"
echo "Archived outputs directories: $ARCHIVED_DIRS"

# ---------------------------------------------------------------------------
# 6. After metrics
# ---------------------------------------------------------------------------
echo ""
echo "=== After cleanup ==="
du -sh tmp/ outputs/ "$ARCHIVE" 2>/dev/null || true
AFTER_COUNT=$(find tmp/ outputs/ -type f 2>/dev/null | wc -l)
echo "Total file count (tmp + outputs): $AFTER_COUNT"

echo ""
echo "=== Cleanup execution completed at $(date -Iseconds) ==="
echo "Review archived files in: $ARCHIVE"
