#!/usr/bin/env bash
# OF-CBM | KTH | Step 5: non-interpretable backbone baseline
#
# Trains a plain linear probe directly on the frozen VideoMAE features (no concept
# bottleneck). This reproduces the "w/o interpretability" row in the paper.
#
# Optional env vars:
#   BACKBONE_FEATS   train feature .pt   (default: features/kth/kth_train_vmae_vit_base_patch16_224.pt)
#   SAVE_DIR         output dir          (default: result/kth_baseline)
#   MODE             only_sparse_cls (GLM-SAGA) or only_cls (CE + Adam)  (default: only_sparse_cls)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

BACKBONE_FEATS="${BACKBONE_FEATS:-features/kth/kth_train_vmae_vit_base_patch16_224.pt}"
SAVE_DIR="${SAVE_DIR:-result/kth_baseline}"
MODE="${MODE:-only_sparse_cls}"

echo "[OF-CBM/KTH] Step 5 | backbone baseline (no CBM)"
echo "  backbone_feats : ${BACKBONE_FEATS}"
echo "  save_dir       : ${SAVE_DIR}"
echo "  mode           : ${MODE}"

python cbm_training/train_of_cbm.py \
    --data_set kth \
    --nb_classes 6 \
    --backbone_features "$BACKBONE_FEATS" \
    --no_cbm \
    --mode_no_cbm "$MODE" \
    --save_dir "$SAVE_DIR" \
    --video_anno_path datasets/KTH \
    --device cuda
