#!/usr/bin/env bash
# OF-CBM | KTH | Step 4 (Patched): CBM training on Patched-flow concepts
#
# Same as 04_train_cbm.sh, but uses the Patched-flow concepts from
# 02_discover_concepts_patch.sh.
#
# Optional env vars:
#   CONCEPT_DIR      discovered-concept dir (default: latest under concept_discovery/output/kth_patch)
#   BACKBONE_FEATS   train feature .pt      (default: features/kth/kth_train_vmae_vit_base_patch16_224.pt)
#   SAVE_DIR         output dir             (default: result/kth_patch)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CONCEPT_DIR="${CONCEPT_DIR:-$(ls -d concept_discovery/output/kth_patch/patch/*/*_trainonly 2>/dev/null | tail -1)}"
if [ -z "${CONCEPT_DIR}" ]; then
    echo "ERROR: concept dir not found. Run 02_discover_concepts_patch.sh first (or set CONCEPT_DIR)." >&2
    exit 1
fi
BACKBONE_FEATS="${BACKBONE_FEATS:-features/kth/kth_train_vmae_vit_base_patch16_224.pt}"
SAVE_DIR="${SAVE_DIR:-result/kth_patch}"

echo "[OF-CBM/KTH] Step 4 | CBM training (Patched)"
echo "  concept_dir    : ${CONCEPT_DIR}"
echo "  backbone_feats : ${BACKBONE_FEATS}"
echo "  save_dir       : ${SAVE_DIR}"

python cbm_training/train_of_cbm.py \
    --data_set kth \
    --nb_classes 6 \
    --backbone_features "$BACKBONE_FEATS" \
    --train_mode pose \
    --pose_label "$CONCEPT_DIR" \
    --loss_mode concept \
    --save_dir "$SAVE_DIR" \
    --video_anno_path datasets/KTH \
    --device cuda
