#!/usr/bin/env bash
set -euo pipefail

# Eval-only script (loads checkpoint and runs evaluation/test path)
# Example:
#   RESUME=/ckpt/1/03-17-2026-14-30/best.pt ./scripts/eval.sh

if [[ -z "${RESUME:-}" ]]; then
  echo "Error: set RESUME to a checkpoint path, e.g. RESUME=/ckpt/1/<time>/best.pt"
  exit 1
fi

SUBJ="${SUBJ:-1}"
TARGET_SUBJ="${TARGET_SUBJ:-$SUBJ}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TRAIN_SPLITS="${TRAIN_SPLITS:-friends-train-default}"
VAL_SPLITS="${VAL_SPLITS:-friends-test-default}"
TEST_SPLITS="${TEST_SPLITS:-movie10-ood-default}"

MODALITY="${MODALITY:-video audio text}"
VIDEO_BACKBONE="${VIDEO_BACKBONE:-metaclip}"
AUDIO_BACKBONE="${AUDIO_BACKBONE:-whisper}"
TEXT_BACKBONE="${TEXT_BACKBONE:-metaclip}"

HIDDEN_DIM="${HIDDEN_DIM:-768}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-1024}"
ENC_LAYERS="${ENC_LAYERS:-0}"
DEC_LAYERS="${DEC_LAYERS:-1}"
NHEADS="${NHEADS:-16}"
NUM_QUERIES="${NUM_QUERIES:-1000}"

accelerate launch main.py \
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
