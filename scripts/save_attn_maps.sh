#!/usr/bin/env bash
set -euo pipefail

# Attention-map export script (saves per-unit HDF5 files)
# Example:
#   bash ./scripts/save_attn_maps.sh

SUBJECTS=(2 3 5 2 3 5)

RESUMES=(
  # "ckpt/1/04-18-2026-01-08/best.pt"
  # "ckpt/2/04-18-2026-01-14/best.pt"
  # "ckpt/3/04-18-2026-22-20/best.pt"
  # "ckpt/5/04-18-2026-22-22/best.pt"

  # "ckpt/1/04-19-2026-02-08/best.pt"
  # "ckpt/2/04-19-2026-02-10/best.pt"
  # "ckpt/3/04-20-2026-00-29/best.pt"
  # "ckpt/5/04-20-2026-00-29/best.pt"

  # "ckpt/1/04-07-2026-16-09/best.pt"
  "ckpt/2/04-07-2026-16-13/best.pt"
  "ckpt/3/04-08-2026-00-52/best.pt"
  "ckpt/5/04-08-2026-00-55/best.pt"
  
  # "ckpt/1/04-08-2026-23-43/best.pt"
  "ckpt/2/04-08-2026-23-48/best.pt"
  "ckpt/3/04-08-2026-23-57/best.pt"
  "ckpt/5/04-12-2026-19-49/best.pt"

)

READOUTS=(
  # "parcels"
  "parcels"
  "parcels"
  "parcels"
  # "voxels"
  "voxels"
  "voxels"
  "voxels"
)

MODALITIES=(video audio text)

VIDEO_BACKBONES=(
  # "videomae"
  # "videomae"
  # "videomae"
  # "videomae"
  # "videomae"
  # "videomae"
  # "videomae"
  # "videomae"
  # "dino"
  # "dino"
  "dino"
  "dino"
  "dino"
  "dino"
  "dino"
  "dino"
)

AUDIO_BACKBONES=(
  # "wav2vec"
  # "wav2vec"
  # "wav2vec"
  # "wav2vec"
  # "wav2vec"
  # "wav2vec"
  # "wav2vec"
  # "wav2vec"
  # "whisper"
  # "whisper"
  "whisper"
  "whisper"
  "whisper"
  "whisper"
  "whisper"
  "whisper"
)

TEXT_BACKBONES=(
  # "deberta"
  # "deberta"
  # "deberta"
  # "deberta"
  # "deberta"
  # "deberta"
  # "deberta"
  # "deberta"
  # "llama"
  # "llama"
  "llama"
  "llama"
  "llama"
  "llama"
  "llama"
  "llama"
)

NUM_WORKERS=4
TEST_SPLITS="movie10-attn-probing"

HIDDEN_DIM=768
DIM_FEEDFORWARD=1024
ENC_LAYERS=0
DEC_LAYERS=1
NHEADS=16
NUM_QUERIES=1000
EXPORT_ROOT="/engram/nklab/pf2477/multimodal_encoder/attn_maps"

EXPECTED_RUNS=${#SUBJECTS[@]}
for name in RESUMES READOUTS VIDEO_BACKBONES AUDIO_BACKBONES TEXT_BACKBONES; do
  eval "count=\${#${name}[@]}"
  if [[ "$count" -ne "$EXPECTED_RUNS" ]]; then
    echo "${name} length mismatch: expected ${EXPECTED_RUNS}, got ${count}" >&2
    exit 1
  fi
done

for i in "${!SUBJECTS[@]}"; do
  subj="${SUBJECTS[$i]}"
  resume="${RESUMES[$i]}"
  run_name="$(basename "$(dirname "$resume")")"
  readout_fmri="${READOUTS[$i]}"
  video_backbone="${VIDEO_BACKBONES[$i]}"
  audio_backbone="${AUDIO_BACKBONES[$i]}"
  text_backbone="${TEXT_BACKBONES[$i]}"

  echo "=== Saving attention maps for subject ${subj} with ${resume} ==="
  pixi run accelerate launch \
    --num_processes 1 \
    --config_file .accelerate/config.yaml \
    --main_process_port 29507 \
    ./attn_map_analysis/save_attn_maps.py \
    --resume "$resume" \
    --subj "$subj" \
    --target_subj "$subj" \
    --batch_size 1 \
    --num_workers "$NUM_WORKERS" \
    --test_splits "$TEST_SPLITS" \
    --attn_maps \
    --attn_write_mode batch \
    --attn_compression lzf \
    --modality "${MODALITIES[@]}" \
    --video_backbone "$video_backbone" \
    --audio_backbone "$audio_backbone" \
    --text_backbone "$text_backbone" \
    --hidden_dim "$HIDDEN_DIM" \
    --dim_feedforward "$DIM_FEEDFORWARD" \
    --enc_layers "$ENC_LAYERS" \
    --dec_layers "$DEC_LAYERS" \
    --nheads "$NHEADS" \
    --num_queries "$NUM_QUERIES" \
    --readout_res "$readout_fmri"

  echo "Saved HDF5 unit files under ${EXPORT_ROOT}/${subj}/${run_name}"
done


while true; do
    echo "keepalive $(date)"
    sleep 7200
done
