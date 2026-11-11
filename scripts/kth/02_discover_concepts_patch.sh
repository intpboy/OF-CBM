#!/usr/bin/env bash
# OF-CBM | KTH | Step 2 (Patched): motion concept discovery (Patched flow)
#
# Same as 02_discover_concepts.sh, but the flow is first divided into a 16x16
# patch grid and each patch is summarized by its mean (dx, dy). This reduces the
# feature dimensionality by ~256x before clustering.
#
# Optional env vars:
#   FLOW_DIR   directory of per-video flow .npy files   (default: flows/kth)
#   OUT_DIR    where to write discovered concepts        (default: concept_discovery/output/kth_patch)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

FLOW_DIR="${FLOW_DIR:-flows/kth}"
OUT_DIR="${OUT_DIR:-concept_discovery/output/kth_patch}"

echo "[OF-CBM/KTH] Step 2 | Patched flow concept discovery"
echo "  flow_dir : ${FLOW_DIR}"
echo "  out_dir  : ${OUT_DIR}"

python -m concept_discovery.motion_discovery.discover_flow_concepts \
    --dataset KTH \
    --flow_path "$FLOW_DIR" \
    --anno_path datasets/KTH \
    --output_path "$OUT_DIR" \
    --approach patch \
    --target_size 224 \
    --target_frames 22 \
    --temporal_sampling center \
    --patch_size 16 \
    --patch_feat_mode dxdy \
    --norm_mode per_sample \
    --clustering_mode partition \
    --use_partition_num -1 \
    --train_only

echo "Concept dir: $(ls -d ${OUT_DIR}/patch/*/*_trainonly 2>/dev/null | tail -1)"
