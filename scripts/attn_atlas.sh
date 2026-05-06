#!/usr/bin/env bash
set -euo pipefail

# Attention-atlas analysis script
# Example:
#   bash ./scripts/attn_atlas.sh

# SUBJECTS=(2 3 5 2 3 5)
SUBJECTS=(
  1
  2
  3
  5

  # 1
  # 2
  # 3
  # 5

  1
  2
  3
  5

  # 1
  # 2
  # 3
  # 5
)

RESUMES=(
  "ckpt/1/04-18-2026-01-08/best.pt"
  "ckpt/2/04-18-2026-01-14/best.pt"
  "ckpt/3/04-18-2026-22-20/best.pt"
  "ckpt/5/04-18-2026-22-22/best.pt"

  # "ckpt/1/04-19-2026-02-08/best.pt"
  # "ckpt/2/04-19-2026-02-10/best.pt"
  # "ckpt/3/04-20-2026-00-29/best.pt"
  # "ckpt/5/04-20-2026-00-29/best.pt"

  "ckpt/1/04-07-2026-16-09/best.pt"
  "ckpt/2/04-07-2026-16-13/best.pt"
  "ckpt/3/04-08-2026-00-52/best.pt"
  "ckpt/5/04-08-2026-00-55/best.pt"

  # "ckpt/1/04-08-2026-23-43/best.pt"
  # "ckpt/2/04-08-2026-23-48/best.pt"
  # "ckpt/3/04-08-2026-23-57/best.pt"
  # "ckpt/5/04-12-2026-19-49/best.pt"
)

READOUTS=(
  "parcels"
  "parcels"
  "parcels"
  "parcels"

  # "voxels"
  # "voxels"
  # "voxels"
  # "voxels"

  "parcels"
  "parcels"
  "parcels"
  "parcels"

  # "voxels"
  # "voxels"
  # "voxels"
  # "voxels"
)

MODALITIES=(video audio text)

VIDEO_BACKBONES=(
  "videomae"
  "videomae"
  "videomae"
  "videomae"

  # "videomae"
  # "videomae"
  # "videomae"
  # "videomae"

  "dino"
  "dino"
  "dino"
  "dino"

  # "dino"
  # "dino"
  # "dino"
  # "dino"
)

AUDIO_BACKBONES=(
  "wav2vec"
  "wav2vec"
  "wav2vec"
  "wav2vec"

  # "wav2vec"
  # "wav2vec"
  # "wav2vec"
  # "wav2vec"

  "whisper"
  "whisper"
  "whisper"
  "whisper"

  # "whisper"
  # "whisper"
  # "whisper"
  # "whisper"
)

TEXT_BACKBONES=(
  "deberta"
  "deberta"
  "deberta"
  "deberta"

  # "deberta"
  # "deberta"
  # "deberta"
  # "deberta"

  "llama"
  "llama"
  "llama"
  "llama"

  # "llama"
  # "llama"
  # "llama"
  # "llama"
)

ATTN_ROOT="/engram/nklab/pf2477/multimodal_encoder/attn_maps"
SAVE_ROOT="./attn_map_analysis/results"
CHUNK_SIZE=1
QC_TOL=5e-3
SAVE_PER_UNIT=0
TIME_WINDOW_SIZE=""
TIME_WINDOW_STRIDE=1

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

  echo "=== Building attention atlas for subject ${subj} | run=${run_name} | readout=${readout_fmri} ==="

  cmd=(
    pixi run python ./attn_map_analysis/parcel_by_modality_attn_atlas.py
    --subject-id "$subj"
    --run-name "$run_name"
    --attn-root "$ATTN_ROOT"
    --save-root "$SAVE_ROOT"
    --modalities "${MODALITIES[@]}"
    --video-backbone "$video_backbone"
    --audio-backbone "$audio_backbone"
    --text-backbone "$text_backbone"
    --chunk-size "$CHUNK_SIZE"
    --qc-tol "$QC_TOL"
    # --top-token-fraction 0.1 0.25 0.5
  )

  if [[ "$SAVE_PER_UNIT" -eq 1 ]]; then
    cmd+=(--save-per-unit)
  fi

  if [[ "$readout_fmri" == "parcels" ]]; then
    cmd+=(--save-token-level)
  else
    cmd+=(--no-save-token-level)
  fi

  if [[ -n "$TIME_WINDOW_SIZE" ]]; then
    cmd+=(--time-window-size "$TIME_WINDOW_SIZE" --time-window-stride "$TIME_WINDOW_STRIDE")
  fi

  "${cmd[@]}"

  echo "Saved atlas outputs under ${SAVE_ROOT}/${subj}/${run_name}"
done


while true; do
    echo "keepalive $(date)"
    sleep 7200
done
