#!/usr/bin/env bash
set -euo pipefail

# Eval-only script (loads checkpoint and runs per-movie Movie10 evaluation)
# Example:
#   bash ./scripts/eval.sh

SUBJECTS=(1 2 3 5)
RESUMES=(

    "ckpt/1/04-07-2026-16-09/best.pt"
    "ckpt/2/04-07-2026-16-13/best.pt"
    "ckpt/3/04-08-2026-00-52/best.pt"
    "ckpt/5/04-08-2026-00-55/best.pt"
)

# SUBJECTS=(5)
# RESUMES=(
#   "ckpt/5/03-31-2026-14-41/best.pt"
# )

MODALITY="video audio text"
VIDEO_BACKBONE="dino"
AUDIO_BACKBONE="whisper"
TEXT_BACKBONE="llama"

BATCH_SIZE=32
NUM_WORKERS=4
TEST_SPLITS="s07"

# Control whether to run per-subject inference before merge.
# RUN_EVAL=0 (default): merge only from existing per-subject outputs.
# RUN_EVAL=1: run inference for each subject, then merge.
RUN_EVAL="${RUN_EVAL:-0}"

# Merged challenge submission output.
MERGE_OUT_DIR="${MERGE_OUT_DIR:-ckpt/challenge}"
MERGE_BASENAME="${MERGE_BASENAME:-fmri_predictions_${TEST_SPLITS}_all_subjects}"

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

if [[ "$RUN_EVAL" == "1" ]]; then
  for i in "${!SUBJECTS[@]}"; do
  subj="${SUBJECTS[$i]}"
  resume="${RESUMES[$i]}"

  echo "=== Evaluating subject ${subj} with ${resume} ==="
  pixi run python test_challenge.py \
    --challenge_split "$TEST_SPLITS" \
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
else
  echo "=== RUN_EVAL=$RUN_EVAL, skipping model inference and merging existing outputs only ==="
fi

echo "=== Merging per-subject predictions into one challenge submission ==="
export SUBJECTS_ENV="$(IFS=,; echo "${SUBJECTS[*]}")"
export RESUMES_ENV="$(printf '%s\n' "${RESUMES[@]}")"
export TEST_SPLITS_ENV="$TEST_SPLITS"
export MERGE_OUT_DIR_ENV="$MERGE_OUT_DIR"
export MERGE_BASENAME_ENV="$MERGE_BASENAME"

pixi run python - <<'PY'
import os
import re
import zipfile
from pathlib import Path

import numpy as np

subjects = [int(x) for x in os.environ["SUBJECTS_ENV"].split(",") if x]
resumes = [x for x in os.environ["RESUMES_ENV"].splitlines() if x]
split = os.environ["TEST_SPLITS_ENV"]
out_dir = Path(os.environ["MERGE_OUT_DIR_ENV"])
base = os.environ["MERGE_BASENAME_ENV"]

if len(subjects) != len(resumes):
  raise RuntimeError(f"subjects/resumes length mismatch: {len(subjects)} vs {len(resumes)}")

allowed_subject_keys = {"sub-01", "sub-02", "sub-03", "sub-05"}
movie_key_pattern = re.compile(r"^s07e\d{2}[a-z]$")
merged = {}

for subj, resume in zip(subjects, resumes):
  subject_key = f"sub-{subj:02d}"
  npy_path = Path(resume).resolve().parent / f"fmri_predictions_{split}_{subject_key}.npy"
  if not npy_path.exists():
    raise FileNotFoundError(f"Missing per-subject prediction file: {npy_path}")

  payload = np.load(npy_path, allow_pickle=True).item()
  if not isinstance(payload, dict):
    raise ValueError(f"{npy_path} must store a dict, got {type(payload)}")
  if subject_key not in payload:
    raise ValueError(f"{npy_path} missing key {subject_key}; found keys: {sorted(payload.keys())}")

  session_dict = payload[subject_key]
  if not isinstance(session_dict, dict):
    raise ValueError(f"{subject_key} value in {npy_path} must be a dict")

  clean_sessions = {}
  for movie_key, arr in session_dict.items():
    if not isinstance(movie_key, str):
      raise ValueError(f"Non-string movie key in {npy_path}: {movie_key!r}")
    if split == "s07" and not movie_key_pattern.match(movie_key):
      raise ValueError(f"Invalid movie key for s07 split in {npy_path}: {movie_key}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
      raise ValueError(f"{subject_key}/{movie_key} must be 2D; got shape={arr.shape}")
    if arr.shape[1] != 1000:
      raise ValueError(f"{subject_key}/{movie_key} must have 1000 parcels; got shape={arr.shape}")

    clean_sessions[movie_key] = arr

  merged[subject_key] = clean_sessions

extra_subject_keys = set(merged.keys()) - allowed_subject_keys
if extra_subject_keys:
  raise ValueError(f"Invalid subject keys: {sorted(extra_subject_keys)}")

out_dir.mkdir(parents=True, exist_ok=True)
npy_out = out_dir / f"{base}.npy"
zip_out = out_dir / f"{base}.zip"

np.save(npy_out, merged)
with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
  zf.write(npy_out, arcname=npy_out.name)

print(f"[Done] merged npy: {npy_out}")
print(f"[Done] merged zip: {zip_out}")
print(f"[Info] subject keys: {sorted(merged.keys())}")
PY
