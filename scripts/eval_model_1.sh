#!/usr/bin/env bash
set -euo pipefail

# Eval-only script (loads checkpoint and runs per-movie Movie10 evaluation)
# Example:
#   bash ./scripts/eval.sh

# SUBJECTS=(1 2 3 5 1 2 3 5 1 2 3 5 1 2 3 5)
SUBJECTS=(1 2 3 5 1 2 3 5)
# SUBJECTS=(1 1 1 1 1 1 1)
# SUBJECTS=(1 2 3 5)

RESUMES=(
    "ckpt/1/04-07-2026-16-09/best.pt" # video audio text
    "ckpt/2/04-07-2026-16-13/best.pt" # video audio text
    "ckpt/3/04-08-2026-00-52/best.pt" # video audio text
    "ckpt/5/04-08-2026-00-55/best.pt" # video audio text
    # "ckpt/1/04-08-2026-13-27/best.pt" # video
    # "ckpt/2/04-08-2026-13-29/best.pt" # video
    # "ckpt/3/04-08-2026-13-49/best.pt" # video
    # "ckpt/5/04-08-2026-19-18/best.pt" # video
    # "ckpt/1/04-08-2026-19-18/best.pt" # audio
    # "ckpt/2/04-08-2026-19-19/best.pt" # audio
    # "ckpt/3/04-08-2026-21-21/best.pt" # audio
    # "ckpt/5/04-08-2026-21-30/best.pt" # audio
    # "ckpt/1/04-09-2026-11-41/best.pt" # text
    # "ckpt/2/04-09-2026-11-43/best.pt" # text
    # "ckpt/3/04-09-2026-11-45/best.pt" # text
    # "ckpt/5/04-09-2026-14-20/best.pt" # text
    "ckpt/1/04-08-2026-23-43/best.pt" # video audio text
    "ckpt/2/04-08-2026-23-48/best.pt" # video audio text
    "ckpt/3/04-08-2026-23-57/best.pt" # video audio text
    "ckpt/5/04-12-2026-19-49/best.pt" # video audio text
    # "ckpt/1/04-09-2026-14-21/best.pt" # video
    # "ckpt/2/04-09-2026-14-52/best.pt" # video
    # "ckpt/3/04-09-2026-16-52/best.pt" # video
    # "ckpt/5/04-09-2026-16-56/best.pt" # video
    # "ckpt/1/04-10-2026-10-50/best.pt" # audio
    # "ckpt/2/04-10-2026-10-51/best.pt" # audio
    # "ckpt/3/04-10-2026-17-12/best.pt" # audio
    # "ckpt/5/04-10-2026-17-13/best.pt" # audio
    # "ckpt/1/04-11-2026-12-44/best.pt" # text
    # "ckpt/2/04-11-2026-12-46/best.pt" # text
    # "ckpt/3/04-11-2026-12-50/best.pt" # text
    # "ckpt/5/04-11-2026-16-23/best.pt" # text
    # "ckpt/1/04-22-2026-16-14/best.pt" # video audio text parcels
    # "ckpt/1/04-22-2026-16-17/best.pt" # video audio text parcels
    # "ckpt/1/04-23-2026-00-25/best.pt" # video audio text parcels
    # "ckpt/1/04-23-2026-00-28/best.pt" # video audio text parcels
    # "ckpt/1/04-23-2026-13-34/best.pt" # video audio text parcels
    # "ckpt/1/04-23-2026-13-35/best.pt" # video audio text parcels
    # "ckpt/1/04-23-2026-23-28/best.pt" # video audio text parcels
    # "ckpt/1/04-24-2026-22-15/best.pt" # video audio text parcels
    # "ckpt/1/04-23-2026-13-37/best.pt" # video audio text voxels
    # "ckpt/1/04-23-2026-23-33/best.pt" # video audio text voxels
    # "ckpt/1/04-23-2026-23-34/best.pt" # video audio text voxels
    # "ckpt/1/04-24-2026-12-08/best.pt" # video audio text voxels
    # "ckpt/1/04-24-2026-12-09/best.pt" # video audio text voxels
    # "ckpt/1/04-24-2026-12-10/best.pt" # video audio text voxels
    # "ckpt/1/04-24-2026-20-05/best.pt" # video audio text voxels
    # "ckpt/1/04-24-2026-20-50/best.pt" # video audio text voxels
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

VIDEO_BACKBONE="dino"
AUDIO_BACKBONE="whisper"
TEXT_BACKBONE="llama"

# READOUT_FMRI="voxels" # "parcels" or "voxels"
# MODALITY="video audio text"

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