#!/usr/bin/env bash
# Prepare and run MVPose on the corrected H36M true-GT protocol.
#
# This script is meant to be launched manually when the local RTX 4090 is free.
# It will exit early if the GPU is busy or if the repo is on the read-only
# A800-D mount.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${SCRIPT_DIR}/mvpose_h36m_config.yaml"

PYTHON=${PYTHON:-python}

cd "${REPO_ROOT}"

# Safety guard: never run on A800-D / read-only mount.
HOSTNAME=$(hostname 2>/dev/null || uname -n)
if [[ "${HOSTNAME}" == "a800-D"* ]] || [[ -d /mnt/nvme0n1/zhangzy/projects ]]; then
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

# 2. Clone MVPose repository if not present.
if [[ ! -d "${MVPose_DIR}/.git" ]]; then
    echo "Cloning MVPose from ${REPO_URL}..."
    mkdir -p "$(dirname "${MVPose_DIR}")"
    git clone "${REPO_URL}" "${MVPose_DIR}"
else
    echo "MVPose repo already cloned at ${MVPose_DIR}."
fi

# 3. Convert common baseline format to MVPose input format.
echo "Converting to MVPose input format..."
"${PYTHON}" "${SCRIPT_DIR}/convert_to_mvpose_format.py" \
    --config "${CONFIG}"

# 4. Run MVPose training / evaluation.
#    The exact launcher varies by upstream repo revision; this calls the
#    common entry point if it exists, otherwise prints next steps.
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
