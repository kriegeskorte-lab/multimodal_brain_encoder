#!/usr/bin/env bash
set -euo pipefail

# Training script for multimodal_encoder/main.py
# Override scalar variables below via env, e.g.:
#   EPOCHS=30 BATCH_SIZE=4 ./scripts/train_baselines.sh

SUBJECTS=(1 2 3 5)
READOUT_FMRIS=(parcels voxels)
EPOCHS=10
BATCH_SIZE=20
NUM_WORKERS=4
LR=1e-4
STEP_SIZE=4
STEP_SIZE_GAMMA=0.5 # do not change for now
WEIGHT_DECAY=5e-3
TRAIN_SPLITS="friends-train-default"
VAL_SPLITS="friends-test-default"
TEST_SPLITS="movie10-ood-default"

MODALITY=(video audio text)
# MODALITY=(text)
VIDEO_BACKBONE="videomae"
AUDIO_BACKBONE="wav2vec"
TEXT_BACKBONE="deberta"
# VIDEO_BACKBONE="dino"
# AUDIO_BACKBONE="whisper"
# TEXT_BACKBONE="llama"
MODALITY_DROPOUT=0.2

HIDDEN_DIM=768
DIM_FEEDFORWARD=1024
ENC_LAYERS=0
DEC_LAYERS=1
NHEADS=16
NUM_QUERIES=1000

USE_WANDB="1"
WANDB_PROJECT="multimodal-encoder"

for READOUT_FMRI in "${READOUT_FMRIS[@]}"; do
  for SUBJ in "${SUBJECTS[@]}"; do
    TARGET_SUBJ="$SUBJ"
    WANDB_RUN_NAME="baseline_sub${SUBJ}_${READOUT_FMRI}"

    if [[ "$USE_WANDB" == "1" ]]; then
      WANDB_FLAGS=(--use_wandb --wandb_project "$WANDB_PROJECT" --wandb_run_name "$WANDB_RUN_NAME")
    else
      WANDB_FLAGS=()
    fi

    echo "=== Training baseline: subject ${SUBJ}, readout ${READOUT_FMRI} ==="
    pixi run accelerate launch \
      --config_file .accelerate/config.yaml \
      --main_process_port 29501 \
      main.py \
      --baseline \
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
      --modality_dropout "$MODALITY_DROPOUT" \
      --readout_res "$READOUT_FMRI" \
      "${WANDB_FLAGS[@]}"
  done
done
