#!/usr/bin/env bash
# OF-CBM | KTH | Step 3: VideoMAE backbone feature extraction
#
# Precomputes frozen VideoMAE (ViT-B/16) features for every KTH video and saves
# them as {FEAT_DIR}/kth_{train,val}_vmae_vit_base_patch16_224.pt. These features
# are the fixed input to the concept projection layer in step 4.
#
# Required env vars:
#   VIDEO_ROOT      directory containing the raw KTH videos
#   FINETUNE_PATH   path to a VideoMAE ViT-B/16 checkpoint (.pth)
# Optional env var:
#   FEAT_DIR        where to save the feature .pt files   (default: features/kth)
#
# The backbone checkpoint is produced separately with the official VideoMAE
# repository (https://github.com/MCG-NJU/VideoMAE); see the README.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

VIDEO_ROOT="${VIDEO_ROOT:?Set VIDEO_ROOT to the directory of raw KTH videos}"
FINETUNE_PATH="${FINETUNE_PATH:?Set FINETUNE_PATH to a VideoMAE ViT-B/16 checkpoint}"
FEAT_DIR="${FEAT_DIR:-features/kth}"
mkdir -p "$FEAT_DIR"

echo "[OF-CBM/KTH] Step 3 | VideoMAE feature extraction"
echo "  video_root    : ${VIDEO_ROOT}"
echo "  finetune_ckpt : ${FINETUNE_PATH}"
echo "  feat_dir      : ${FEAT_DIR}"

python cbm_training/feature_extraction/save_features.py \
    --data_set kth \
    --nb_classes 6 \
    --backbone vmae_vit_base_patch16_224 \
    --finetune "$FINETUNE_PATH" \
    --data_path "$VIDEO_ROOT" \
    --video_anno_path datasets/KTH \
    --batch_size 32 \
    --num_frames 16 \
    --device cuda \
    --activation_dir "$FEAT_DIR"
