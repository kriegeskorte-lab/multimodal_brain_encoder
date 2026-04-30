#!/usr/bin/env bash
set -euo pipefail

# Causal eval for videomae + wav2vec + deberta checkpoints.
# For each full three-modality checkpoint, rerun inference with one modality
# removed from --modality and save to test_causal_movie_breakdown.json.

SUBJECTS=(1 2 3 5)
READOUTS=(voxels parcels)
MODALITIES=(video audio text)

VIDEO_BACKBONE="videomae"
AUDIO_BACKBONE="wav2vec"
TEXT_BACKBONE="deberta"

BATCH_SIZE=16
NUM_WORKERS=4
TEST_SPLITS="movie10-ood-default"

HIDDEN_DIM=768
DIM_FEEDFORWARD=1024
ENC_LAYERS=0
DEC_LAYERS=1
NHEADS=16
NUM_QUERIES=1000

checkpoint_for() {
  local subj="$1"
  local readout="$2"

  case "${subj}:${readout}" in
    1:parcels) echo "ckpt/1/04-18-2026-01-08/best.pt" ;;
    2:parcels) echo "ckpt/2/04-18-2026-01-14/best.pt" ;;
    3:parcels) echo "ckpt/3/04-18-2026-22-20/best.pt" ;;
    5:parcels) echo "ckpt/5/04-18-2026-22-22/best.pt" ;;
    1:voxels) echo "ckpt/1/04-19-2026-02-08/best.pt" ;;
    2:voxels) echo "ckpt/2/04-19-2026-02-10/best.pt" ;;
    3:voxels) echo "ckpt/3/04-20-2026-00-29/best.pt" ;;
    5:voxels) echo "ckpt/5/04-20-2026-00-29/best.pt" ;;
    *)
      echo "No checkpoint configured for subject=${subj}, readout=${readout}" >&2
      return 1
      ;;
  esac
}

kept_modalities_after_masking() {
  local removed="$1"

  case "$removed" in
    video) echo "audio text" ;;
    audio) echo "video text" ;;
    text) echo "video audio" ;;
    *)
      echo "Unsupported modality to remove: ${removed}" >&2
      return 1
      ;;
  esac
}

for subj in "${SUBJECTS[@]}"; do
  for readout in "${READOUTS[@]}"; do
    resume="$(checkpoint_for "$subj" "$readout")"

    for removed_modality in "${MODALITIES[@]}"; do
      kept_modality_string="$(kept_modalities_after_masking "$removed_modality")"
      read -r -a kept_modalities <<< "$kept_modality_string"

      echo "=== Causal eval subject=${subj} readout=${readout} remove=${removed_modality} resume=${resume} ==="
      pixi run accelerate launch \
        --config_file .accelerate/config.yaml --main_process_port 29511 \
        test.py \
        --resume "$resume" \
        --subj "$subj" \
        --target_subj "$subj" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --test_splits "$TEST_SPLITS" \
        --modality "${kept_modalities[@]}" \
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
        --save_test_causal_intervention
    done
  done
done
