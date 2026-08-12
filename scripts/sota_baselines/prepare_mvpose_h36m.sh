#!/usr/bin/env bash
# Prepare and run MVPose on the corrected H36M true-GT protocol.
#
# This script is meant to be launched manually when the local RTX 4090 is free.
# It will exit early if the GPU is busy or if the repo is on the read-only
# A800-D mount.
#
# IMPORTANT: The zju3dv/mvpose repo is reachable, but running the full
# detector/Re-ID backend on H36M is not required. The H36M true-GT evaluation
# uses scripts/sota_baselines/mvpose_h36m_adapter.py, which bypasses the 2D
# detector and drives the geometry-only top-down triangulation kernel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${SCRIPT_DIR}/mvpose_h36m_config.yaml"

PYTHON=${PYTHON:-python}

cd "${REPO_ROOT}"

# Safety guard: never run on A800-D / read-only mount.
HOSTNAME=$(hostname 2>/dev/null || uname -n)
if [[ "${HOSTNAME}" == "a800-D"* ]] || [[ -d /mnt/nvme0n1p1/zhangzy/projects ]]; then
    echo "ERROR: This script must not run on A800-D (read-only)." >&2
    exit 1
fi

# Safety guard: do not start if the GPU is busy.
if ! bash "${SCRIPT_DIR}/check_gpu_free.sh"; then
    echo "GPU is not free. Aborting." >&2
    exit 1
fi

# Load helper values from the YAML config.
MVPose_DIR=$(python - <<'PY'
import yaml
with open("scripts/sota_baselines/mvpose_h36m_config.yaml") as f:
    cfg = yaml.safe_load(f)
print(cfg["repo"]["dir"])
PY
)
REPO_URL=$(python - <<'PY'
import yaml
with open("scripts/sota_baselines/mvpose_h36m_config.yaml") as f:
    cfg = yaml.safe_load(f)
print(cfg["repo"]["url"])
PY
)
INPUT_PKL=$(python - <<'PY'
import yaml
with open("scripts/sota_baselines/mvpose_h36m_config.yaml") as f:
    cfg = yaml.safe_load(f)
print(cfg["input_pkl"])
PY
)

# 1. Export H36M true-GT to the common baseline format if needed.
if [[ ! -f "${INPUT_PKL}" ]]; then
    echo "Exporting H36M true-GT to common baseline format..."
    "${PYTHON}" "${SCRIPT_DIR}/common_export_h36m_true_gt.py"
fi

# 2. Check / clone MVPose repository.
if [[ -d "${MVPose_DIR}/.git" ]]; then
    echo "MVPose repo already cloned at ${MVPose_DIR}."
    MVPose_READY=true
elif [[ ! -d "${MVPose_DIR}" ]]; then
    echo "Cloning zju3dv/mvpose into ${MVPose_DIR}..."
    mkdir -p "$(dirname "${MVPose_DIR}")"
    git clone --depth 1 "${REPO_URL}" "${MVPose_DIR}"
    MVPose_READY=true
else
    echo "WARN: ${MVPose_DIR} exists but does not look like the MVPose repo." >&2
    MVPose_READY=false
fi

# 3. Convert common baseline format to MVPose input format.
echo "Converting to MVPose input format..."
"${PYTHON}" "${SCRIPT_DIR}/convert_to_mvpose_format.py" \
    --config "${CONFIG}"

# 4. Run MVPose training / evaluation only if the upstream repo is present.
if [[ "${MVPose_READY}" != "true" ]]; then
    echo ""
    echo "MVPose data prepared, but upstream repository is missing." >&2
    echo "Next steps:" >&2
    echo "  1. Clone a working MVPose fork into ${MVPose_DIR}" >&2
    echo "  2. Wire its loader to the manifest:" >&2
    echo "     $(python -c 'import yaml; print(yaml.safe_load(open("scripts/sota_baselines/mvpose_h36m_config.yaml"))["data_dir"])')/manifest.json" >&2
    echo "  3. Then re-run this script." >&2
    exit 0
fi

# Best-effort launch.
TRAIN_PY="${MVPose_DIR}/train.py"
TEST_PY="${MVPose_DIR}/test.py"
CHECKPOINT="${MVPose_DIR}/output/h36m_true_gt/best.pth"

if [[ -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint exists; running evaluation."
    if [[ -f "${TEST_PY}" ]]; then
        "${PYTHON}" "${TEST_PY}" --cfg "${CONFIG}"
    else
        echo "WARN: ${TEST_PY} not found. Please run MVPose evaluation manually." >&2
    fi
else
    echo "Starting MVPose training..."
    if [[ -f "${TRAIN_PY}" ]]; then
        "${PYTHON}" "${TRAIN_PY}" --cfg "${CONFIG}"
    else
        echo "WARN: ${TRAIN_PY} not found. Please run MVPose training manually." >&2
    fi
fi

echo "MVPose prep/run complete. See outputs/sota_baselines/ for logs/metrics."
