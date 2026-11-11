#!/usr/bin/env bash
# OF-CBM | Penn Action | Step 3: VideoMAE backbone feature extraction
#
# Precomputes frozen VideoMAE (ViT-B/16) features for every Penn Action video and
# saves them as {FEAT_DIR}/penn-action_{train,val}_vmae_vit_base_patch16_224.pt.
#
# Required env vars:
#   VIDEO_ROOT      directory containing the raw Penn Action videos
#   FINETUNE_PATH   path to a VideoMAE ViT-B/16 checkpoint (.pth)
# Optional env var:
#   FEAT_DIR        where to save the feature .pt files   (default: features/penn_action)
#
# The backbone checkpoint is produced separately with the official VideoMAE
# repository (https://github.com/MCG-NJU/VideoMAE); see the README.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

VIDEO_ROOT="${VIDEO_ROOT:?Set VIDEO_ROOT to the directory of raw Penn Action videos}"
FINETUNE_PATH="${FINETUNE_PATH:?Set FINETUNE_PATH to a VideoMAE ViT-B/16 checkpoint}"
FEAT_DIR="${FEAT_DIR:-features/penn_action}"
mkdir -p "$FEAT_DIR"

echo "[OF-CBM/Penn] Step 3 | VideoMAE feature extraction"
echo "  video_root    : ${VIDEO_ROOT}"
echo "  finetune_ckpt : ${FINETUNE_PATH}"
echo "  feat_dir      : ${FEAT_DIR}"

python cbm_training/feature_extraction/save_features.py \
    --data_set penn-action \
    --nb_classes 15 \
    --backbone vmae_vit_base_patch16_224 \
    --finetune "$FINETUNE_PATH" \
    --data_path "$VIDEO_ROOT" \
    --video_anno_path datasets/Penn_action \
    --batch_size 32 \
    --num_frames 16 \
    --device cuda \
    --activation_dir "$FEAT_DIR"
