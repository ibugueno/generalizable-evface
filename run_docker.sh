#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-facial-analysis:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-facial-analysis}"
GPU_ID="${GPU_ID:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${GPU_ID}" != "0" ]]; then
    echo "This first implementation is restricted to host GPU 0 (GPU_ID=0)." >&2
    exit 1
fi

# Set FACIAL_ANALYSIS_PERSISTENT_DIR on the server to the large persistent
# volume. The local default is deliberately inside this project and contains
# no source data until the user places it there.
PERSISTENT_DIR="${FACIAL_ANALYSIS_PERSISTENT_DIR:-${SCRIPT_DIR}/persistent}"
DATA_DIR="${FACIAL_ANALYSIS_DATA_DIR:-${PERSISTENT_DIR}/data}"
FACEPTOR_DIR="${FACIAL_ANALYSIS_FACEPTOR_DIR:-${PERSISTENT_DIR}/faceptor}"

mkdir -p "${PERSISTENT_DIR}" "${DATA_DIR}" "${FACEPTOR_DIR}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run --gpus "device=${GPU_ID}" -it \
    --name "${CONTAINER_NAME}" \
    --shm-size="${SHM_SIZE:-8g}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e FACEPTOR_CONFIG="${FACEPTOR_CONFIG:-}" \
    -e PYTHONPATH=/app/facial-analysis:/app/faceptor/faceptor_project \
    -v "${SCRIPT_DIR}:/app/facial-analysis:ro" \
    -v "${PERSISTENT_DIR}:/app/persistent" \
    -v "${DATA_DIR}:/app/data:ro" \
    -v "${FACEPTOR_DIR}:/app/faceptor:ro" \
    "${IMAGE_NAME}" bash
