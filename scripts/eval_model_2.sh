#!/usr/bin/env bash
set -euo pipefail

# Eval-only script (loads checkpoint and runs per-movie Movie10 evaluation)
# Example:
#   bash ./scripts/eval.sh

# SUBJECTS=(1 2 3 5 1 2 3 5 1 2 3 5 1 2 3 5)
SUBJECTS=(1 2 3 5 1 2 3 5)
# SUBJECTS=(1)

RESUMES=(
  "ckpt/1/04-18-2026-01-08/best.pt" # video audio text
  "ckpt/2/04-18-2026-01-14/best.pt" # video audio text
  "ckpt/3/04-18-2026-22-20/best.pt" # video audio text
  "ckpt/5/04-18-2026-22-22/best.pt" # video audio text
  # "ckpt/1/04-15-2026-19-12/best.pt" # video
  # "ckpt/2/04-15-2026-19-10/best.pt" # video
  # "ckpt/3/04-15-2026-19-10/best.pt" # video 
  # "ckpt/5/04-16-2026-13-45/best.pt" # video
  # "ckpt/1/04-16-2026-13-46/best.pt" # audio
  # "ckpt/2/04-16-2026-13-46/best.pt" # audio
  # "ckpt/3/04-16-2026-23-22/best.pt" # audio
  # "ckpt/5/04-17-2026-13-49/best.pt" # audio
  # "ckpt/1/04-17-2026-11-20/best.pt" # text
  # "ckpt/2/04-17-2026-13-53/best.pt" # text
  # "ckpt/3/04-17-2026-15-17/best.pt" # text
  # "ckpt/5/04-17-2026-18-52/best.pt" # text
  "ckpt/1/04-19-2026-02-08/best.pt" # video audio text
  "ckpt/2/04-19-2026-02-10/best.pt" # video audio text
  "ckpt/3/04-20-2026-00-29/best.pt" # video audio text
  "ckpt/5/04-20-2026-00-29/best.pt" # video audio text
  # "ckpt/1/04-20-2026-04-03/best.pt" # video
  # "ckpt/2/04-20-2026-04-03/best.pt" # video
  # "ckpt/3/04-20-2026-12-49/best.pt" # video
  # "ckpt/5/04-20-2026-12-52/best.pt" # video
  # "ckpt/1/04-20-2026-23-34/best.pt" # audio
  # "ckpt/2/04-21-2026-00-50/best.pt" # audio
  # "ckpt/3/04-22-2026-11-32/best.pt" # audio
  # "ckpt/5/04-21-2026-15-23/best.pt" # audio
  # "ckpt/1/04-21-2026-22-19/best.pt" # text
  # "ckpt/2/04-21-2026-22-20/best.pt" # text
  # "ckpt/3/04-22-2026-01-13/best.pt" # text
  # "ckpt/5/04-22-2026-11-33/best.pt" # text

)

MODALITIES=(
  "video audio text"
  "video audio text"
  "video audio text"
  "video audio text"
  "video audio text"
  "video audio text"
  "video audio text"
  "video audio text"
)

READOUTS=(
  "parcels"
  "parcels"
  "parcels"
  "parcels"
  "voxels"
  "voxels"
  "voxels"
  "voxels"
)

VIDEO_BACKBONE="videomae"
AUDIO_BACKBONE="wav2vec"
TEXT_BACKBONE="deberta"

# READOUT_FMRI="voxels" # "parcels" or "voxels"
# MODALITY="video audio text"

BATCH_SIZE=8
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
  readout="${READOUTS[$i]}"
  mod="${MODALITIES[$i]}"
  read -r -a modality <<< "$mod"
  echo "modality tokens: ${modality[*]}"

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
    --modality "${modality[@]}" \
    --video_backbone "$VIDEO_BACKBONE" \
    --audio_backbone "$AUDIO_BACKBONE" \
    --text_backbone "$TEXT_BACKBONE" \
    --hidden_dim "$HIDDEN_DIM" \
    --dim_feedforward "$DIM_FEEDFORWARD" \
    --enc_layers "$ENC_LAYERS" \
    --dec_layers "$DEC_LAYERS" \
    --nheads "$NHEADS" \
    --num_queries "$NUM_QUERIES" \
    --readout_res "$readout" \
    --save_encoding_acc \
    --save_test_movie_breakdown
done