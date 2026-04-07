#!/usr/bin/env bash
set -euo pipefail

# Sanity check script: tiny train/val/test loop, lr forced to 0 in main.py,
# wandb/checkpoint disabled by pipeline_sanity_check mode.
# Example:
#   SUBJ=1 SANITY_BATCHES=2 ./scripts/sanity_check.sh

SUBJ=1
TARGET_SUBJ=1
BATCH_SIZE=4
NUM_WORKERS=0
EPOCHS=5
SANITY_BATCHES=8

HIDDEN_DIM=256
DIM_FEEDFORWARD=512
ENC_LAYERS=0
DEC_LAYERS=1
NHEADS=8
NUM_QUERIES=1000

MODALITY=(video audio text)
VIDEO_BACKBONE="metaclip"
AUDIO_BACKBONE="whisper"
TEXT_BACKBONE="metaclip"

# bash scripts/sanity_check.sh
pixi run accelerate launch \
  --config_file .accelerate/config.yaml \
  main.py \
  --pipeline_sanity_check \
  --subj "$SUBJ" \
  --target_subj "$TARGET_SUBJ" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --sanity_batches "$SANITY_BATCHES" \
  --hidden_dim "$HIDDEN_DIM" \
  --dim_feedforward "$DIM_FEEDFORWARD" \
  --enc_layers "$ENC_LAYERS" \
  --dec_layers "$DEC_LAYERS" \
  --nheads "$NHEADS" \
  --num_queries "$NUM_QUERIES" \
  --modality "${MODALITY[@]}" \
  --video_backbone "$VIDEO_BACKBONE" \
  --audio_backbone "$AUDIO_BACKBONE" \
  --text_backbone "$TEXT_BACKBONE" \
