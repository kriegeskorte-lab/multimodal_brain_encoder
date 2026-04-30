#!/usr/bin/env bash
set -euo pipefail

# Attention-map export script (saves per-unit HDF5 files)
# Example:
#   bash ./scripts/save_attn_maps.sh

SUBJECTS=(1 2 3 5)
# RESUMES=(
#   "ckpt/1/04-07-2026-16-09/best.pt"
#   "ckpt/2/04-07-2026-16-13/best.pt"
#   "ckpt/3/04-08-2026-00-52/best.pt"
#   "ckpt/5/04-08-2026-00-55/best.pt"
# )
# READOUT_FMRI="parcels"

RESUMES=(
  "ckpt/1/04-08-2026-23-43/best.pt"
  "ckpt/2/04-08-2026-23-48/best.pt"
  "ckpt/3/04-08-2026-23-57/best.pt"
  "ckpt/5/04-12-2026-19-49/best.pt"
)
READOUT_FMRI="voxels"

MODALITY="video audio text"
VIDEO_BACKBONE="dino"
AUDIO_BACKBONE="whisper"
TEXT_BACKBONE="llama"

NUM_WORKERS=4
TEST_SPLITS="movie10-attn-probing"

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
  run_name="$(basename "$(dirname "$resume")")"

  echo "=== Saving attention maps for subject ${subj} with ${resume} ==="
  pixi run accelerate launch \
    --num_processes 1 \
    --config_file .accelerate/config.yaml \
    --main_process_port 29507 \
    ./attn_map_analysis/save_attn_maps.py \
    --resume "$resume" \
    --subj "$subj" \
    --target_subj "$subj" \
    --batch_size 4 \
    --num_workers "$NUM_WORKERS" \
    --test_splits "$TEST_SPLITS" \
    --attn_maps \
    --attn_write_mode batch \
    --attn_compression lzf \
    --modality $MODALITY \
    --video_backbone "$VIDEO_BACKBONE" \
    --audio_backbone "$AUDIO_BACKBONE" \
    --text_backbone "$TEXT_BACKBONE" \
    --hidden_dim "$HIDDEN_DIM" \
    --dim_feedforward "$DIM_FEEDFORWARD" \
    --enc_layers "$ENC_LAYERS" \
    --dec_layers "$DEC_LAYERS" \
    --nheads "$NHEADS" \
    --num_queries "$NUM_QUERIES" \
    --readout_res "$READOUT_FMRI"

  echo "Saved HDF5 unit files under ./attn_maps/${subj}/${run_name}"
done
