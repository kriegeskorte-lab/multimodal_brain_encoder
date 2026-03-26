#!/usr/bin/env bash
set -euo pipefail

# Eval-only script (loads checkpoint and runs evaluation/test path)
# Example:
#   bash ./scripts/eval.sh

RESUME="ckpt/1/03-18-2026-02-20/best.pt" 

SUBJ=1
TARGET_SUBJ=1
BATCH_SIZE=32
NUM_WORKERS=4
TRAIN_SPLITS="friends-train-default"
VAL_SPLITS="friends-test-default"
TEST_SPLITS="movie10-ood-default"

MODALITY="video audio text"
VIDEO_BACKBONE="metaclip"
AUDIO_BACKBONE="whisper"
TEXT_BACKBONE="metaclip"

HIDDEN_DIM=256
DIM_FEEDFORWARD=1024
ENC_LAYERS=0
DEC_LAYERS=4
NHEADS=8
NUM_QUERIES=1000

pixi run accelerate launch \
  --config_file .accelerate/config.yaml --main_process_port 0\
  main.py \
  --eval_only \
  --resume "$RESUME" \
  --subj "$SUBJ" \
  --target_subj "$TARGET_SUBJ" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --train_splits "$TRAIN_SPLITS" \
  --val_splits "$VAL_SPLITS" \
  --test_splits "$TEST_SPLITS" \
  --modality $MODALITY \
  --video_backbone "$VIDEO_BACKBONE" \
  --audio_backbone "$AUDIO_BACKBONE" \
  --text_backbone "$TEXT_BACKBONE" \
  --hidden_dim "$HIDDEN_DIM" \
  --dim_feedforward "$DIM_FEEDFORWARD" \
  --enc_layers "$ENC_LAYERS" \
  --dec_layers "$DEC_LAYERS" \
  --nheads "$NHEADS" \
  --num_queries "$NUM_QUERIES"
