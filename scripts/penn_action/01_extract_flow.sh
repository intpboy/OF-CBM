#!/usr/bin/env bash
# OF-CBM | Penn Action | Step 1: dense optical flow extraction
#
# Extracts dense optical flow for every video listed in
# datasets/Penn_action/{train,val}.csv using a pretrained RAFT model
# (torchvision). One .npy array of shape (L-1, H, W, 2) is written per video.
#
# The flow method is pluggable: the paper used WAFT, but any dense-flow model
# works as long as it produces the same per-video (L-1, H, W, 2) .npy layout.
#
# Required env var:
#   VIDEO_ROOT   directory containing the raw Penn Action videos
# Optional env var:
#   FLOW_DIR     where to write the .npy flow arrays   (default: flows/penn_action)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

VIDEO_ROOT="${VIDEO_ROOT:?Set VIDEO_ROOT to the directory of raw Penn Action videos}"
FLOW_DIR="${FLOW_DIR:-flows/penn_action}"
mkdir -p "$FLOW_DIR"

echo "[OF-CBM/Penn] Step 1 | RAFT flow extraction"
echo "  video_root : ${VIDEO_ROOT}"
echo "  flow_dir   : ${FLOW_DIR}"

python flow_extraction/extract_raft_flow.py \
    --video_root "$VIDEO_ROOT" \
    --anno_path datasets/Penn_action \
    --output_dir "$FLOW_DIR" \
    --resize 224 224 \
    --dtype float16 \
    --skip_existing
