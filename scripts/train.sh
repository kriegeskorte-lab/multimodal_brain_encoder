#!/usr/bin/env bash
set -euo pipefail

# Training script for multimodal_encoder/main.py
# Override any variable below via env, e.g.:
#   SUBJ=2 EPOCHS=30 BATCH_SIZE=4 ./scripts/train.sh

SUBJ=1
TARGET_SUBJ=1
EPOCHS=100
BATCH_SIZE=64
NUM_WORKERS=4
LR=1e-4
STEP_SIZE=30
STEP_SIZE_GAMMA=0.5
WEIGHT_DECAY=1e-4
TRAIN_SPLITS="friends-train-default"
VAL_SPLITS="friends-test-default"
TEST_SPLITS="movie10-ood-default"

MODALITY=(video audio text)
VIDEO_BACKBONE="metaclip"
AUDIO_BACKBONE="whisper"
TEXT_BACKBONE="metaclip"

HIDDEN_DIM=256
DIM_FEEDFORWARD=1024
ENC_LAYERS=0
DEC_LAYERS=4
NHEADS=8
NUM_QUERIES=1000

USE_WANDB="1"
WANDB_PROJECT="multimodal-encoder"
WANDB_RUN_NAME="train-sub${SUBJ}"

if [[ "$USE_WANDB" == "1" ]]; then
  WANDB_FLAGS=(--use_wandb --wandb_project "$WANDB_PROJECT" --wandb_run_name "$WANDB_RUN_NAME")
else
  WANDB_FLAGS=()
fi

pixi run accelerate launch \
  --config_file .accelerate/config.yaml \
  main.py \
  --subj "$SUBJ" \
  --target_subj "$TARGET_SUBJ" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --step_size "$STEP_SIZE" \
  --step_size_gamma "$STEP_SIZE_GAMMA" \
  --weight_decay "$WEIGHT_DECAY" \
  --train_splits "$TRAIN_SPLITS" \
  --val_splits "$VAL_SPLITS" \
  --test_splits "$TEST_SPLITS" \
  --modality "${MODALITY[@]}" \
  --video_backbone "$VIDEO_BACKBONE" \
  --audio_backbone "$AUDIO_BACKBONE" \
  --text_backbone "$TEXT_BACKBONE" \
  --hidden_dim "$HIDDEN_DIM" \
  --dim_feedforward "$DIM_FEEDFORWARD" \
  --enc_layers "$ENC_LAYERS" \
  --dec_layers "$DEC_LAYERS" \
  --nheads "$NHEADS" \
  --num_queries "$NUM_QUERIES" \
  "${WANDB_FLAGS[@]}"
