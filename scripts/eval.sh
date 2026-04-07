#!/usr/bin/env bash
set -euo pipefail

# Eval-only script (loads checkpoint and runs per-movie Movie10 evaluation)
# Example:
#   bash ./scripts/eval.sh

# SUBJECTS=(1 2 3 5)
# RESUMES=(
#   "ckpt/1/03-27-2026-17-39/best.pt"
#   "ckpt/2/03-27-2026-19-33/best.pt"
#   "ckpt/3/03-28-2026-00-53/best.pt"
#   "ckpt/5/03-28-2026-11-19/best.pt"
# )

# SUBJECTS=(1 2 3)
# RESUMES=(
#   "ckpt/1/03-30-2026-11-22/best.pt"
#   "ckpt/2/03-30-2026-23-29/best.pt"
#   "ckpt/3/03-30-2026-23-32/best.pt"
# )

SUBJECTS=(5)
RESUMES=(
  "ckpt/5/03-31-2026-14-41/best.pt"
)

MODALITY="video audio text"
VIDEO_BACKBONE="dino"
AUDIO_BACKBONE="whisper"
TEXT_BACKBONE="llama"

BATCH_SIZE=32
NUM_WORKERS=4
TEST_SPLITS="movie10-ood-default"

HIDDEN_DIM=768
DIM_FEEDFORWARD=1024
ENC_LAYERS=0
DEC_LAYERS=1
NHEADS=16
NUM_QUERIES=1000

if [[ ${#SUBJECTS[@]} -ne ${#RESUMES[@]} ]]; then
  echo "SUBJECTS and RESUMES length mismatch" >&2
  exit 1
fi

for i in "${!SUBJECTS[@]}"; do
  subj="${SUBJECTS[$i]}"
  resume="${RESUMES[$i]}"

  echo "=== Evaluating subject ${subj} with ${resume} ==="
  pixi run accelerate launch \
    --config_file .accelerate/config.yaml --main_process_port 29506 \
    test.py \
    --resume "$resume" \
    --subj "$subj" \
    --target_subj "$subj" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
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
done
