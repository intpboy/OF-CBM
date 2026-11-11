#!/usr/bin/env bash
# OF-CBM | Penn Action | Step 2: motion concept discovery (Raw flow)
#
# Clusters the normalized, flattened flow of each training video with FINCH and
# selects the partition with the highest NMI against the action labels. Each
# cluster is represented by its medoid; validation videos are assigned to the
# nearest medoid (--train_only) to avoid label leakage.
#
# Optional env vars:
#   FLOW_DIR   directory of per-video flow .npy files   (default: flows/penn_action)
#   OUT_DIR    where to write discovered concepts        (default: concept_discovery/output/penn_action_raw)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

FLOW_DIR="${FLOW_DIR:-flows/penn_action}"
OUT_DIR="${OUT_DIR:-concept_discovery/output/penn_action_raw}"

echo "[OF-CBM/Penn] Step 2 | Raw flow concept discovery"
echo "  flow_dir : ${FLOW_DIR}"
echo "  out_dir  : ${OUT_DIR}"

python -m concept_discovery.motion_discovery.discover_flow_concepts \
    --dataset Penn_action \
    --flow_path "$FLOW_DIR" \
    --anno_path datasets/Penn_action \
    --output_path "$OUT_DIR" \
    --approach raw \
    --target_size 224 \
    --target_frames 30 \
    --temporal_sampling center \
    --clustering_mode partition \
    --use_partition_num -1 \
    --train_only

echo "Concept dir: $(ls -d ${OUT_DIR}/raw/raw/*_trainonly 2>/dev/null | tail -1)"
