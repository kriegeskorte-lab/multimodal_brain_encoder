from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import h5py
import numpy as np
from tqdm.auto import tqdm

'''
pixi run python ./attn_map_analysis/parcel_by_modality_attn_atlas.py \
  --subject-id 1 \
  --run-name 04-18-2026-01-08 \
  --video-backbone videomae \
  --audio-backbone wav2vec \
  --text-backbone deberta

pixi run python ./attn_map_analysis/parcel_by_modality_attn_atlas.py \
  --subject-id 1 \
  --run-name 04-19-2026-02-08 \
  --video-backbone videomae \
  --audio-backbone wav2vec \
  --text-backbone deberta

pixi run python ./attn_map_analysis/parcel_by_modality_attn_atlas.py \
  --subject-id 1 \
  --run-name 04-07-2026-16-09 \
  --video-backbone dino \
  --audio-backbone whisper \
  --text-backbone llama

pixi run python ./attn_map_analysis/parcel_by_modality_attn_atlas.py \
  --subject-id 1 \
  --run-name 04-08-2026-23-43 \
  --video-backbone dino \
  --audio-backbone whisper \
  --text-backbone llama
'''


DEFAULT_RUN_BY_SUBJECT = {
    1: "04-18-2026-01-08", # parcel
    # 1: "04-19-2026-02-08", # voxel
    # 1: "04-07-2026-16-09", # parcel
    # 1: "04-08-2026-23-43", # voxel
    # 2: "04-07-2026-16-13", # parcel
    # 3: "04-08-2026-00-52", # parcel
    # 5: "04-08-2026-00-55", # parcel
    # 1: "04-08-2026-23-43", # voxel
    # 2: "04-08-2026-23-48", # voxel
    # 3: "04-08-2026-23-57", # voxel
    # 5: "04-12-2026-19-49", # voxel
}

MODALITY_ORDER = ("video", "audio", "text")

VIDEO_TOKEN_COUNTS = {
    "dino": 16 * 201,
    "videomae": 1568,
    "timesformer": 3137,
    "metaclip": 16 * 197,
    "openaiclip": 16 * 257,
}

AUDIO_TOKEN_COUNTS = {
    "wav2vec": 1539,
    "whisper": 1500,
}

TEXT_TOKEN_COUNTS = {
    "llama": 512,
    "deberta": 512,
    "metaclip": 77,
    "openaiclip": 77,
}


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    backbone: str
    start: int
    end: int

    @property
    def num_tokens(self) -> int:
        return self.end - self.start


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 1: parcel-by-modality attention atlas from saved HDF5 attention maps."
    )
    parser.add_argument("--subject-id", type=int, default=1)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--attn-root", type=Path, default=Path("./attn_maps"))
    parser.add_argument("--save-root", type=Path, default=Path("./attn_map_analysis/results"))
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=list(MODALITY_ORDER),
        choices=list(MODALITY_ORDER),
        help="Subset of modalities present in the saved attention tensor.",
    )
    parser.add_argument("--video-backbone", type=str, default="dino")
    parser.add_argument("--audio-backbone", type=str, default="whisper")
    parser.add_argument("--text-backbone", type=str, default="llama")
    parser.add_argument(
        "--units",
        nargs="*",
        default=None,
        help="Optional explicit unit names. Defaults to manifest units if available.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Number of samples to read per HDF5 chunk.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for debugging; if omitted, process all saved samples.",
    )
    parser.add_argument(
        "--qc-tol",
        type=float,
        default=5e-3,
        help="Tolerance for checking whether attention sums to 1 across memory tokens.",
    )
    parser.add_argument(
        "--save-per-unit",
        action="store_true",
        help="Debug mode: also save combined-style outputs for each unit shard.",
    )
    parser.add_argument(
        "--save-token-level",
        dest="save_token_level",
        action="store_true",
        default=True,
        help=(
            "Save token-level parcel attention outputs [Q, T]. This is storage-feasible for parcel "
            "readouts but can be large for voxel readouts."
        ),
    )
    parser.add_argument(
        "--no-save-token-level",
        dest="save_token_level",
        action="store_false",
        help="Disable token-level [Q, T] outputs while keeping modality-level summaries.",
    )
    parser.add_argument(
        "--time-window-size",
        type=int,
        default=None,
        help=(
            "Optional sliding-window size in samples/TRs within each split. "
            "If provided, save per-window attention dynamics."
        ),
    )
    parser.add_argument(
        "--time-window-stride",
        type=int,
        default=1,
        help="Stride in samples/TRs for sliding time windows.",
    )
    parser.add_argument(
        "--top-token-fraction",
        nargs="+",
        default=["0.25"],
        help=(
            "One or more fractions of tokens within each modality used for robust top-token preference maps. "
            "Accepts space-separated values such as '0.1 0.25 0.5' or comma-separated values. "
            "For each parcel/query, the script averages the highest-scoring tokens within each modality, "
            "then normalizes those modality scores across modalities."
        ),
    )
    return parser


def normalize_modalities(modalities: Sequence[str]) -> List[str]:
    normalized = [str(modality).strip().lower() for modality in modalities if str(modality).strip()]
    invalid = [modality for modality in normalized if modality not in MODALITY_ORDER]
    if invalid:
        raise ValueError(f"Unsupported modalities: {invalid}")
    return [modality for modality in MODALITY_ORDER if modality in normalized]


def normalize_top_token_fractions(values: Sequence[str | float]) -> List[float]:
    fractions: List[float] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            fraction = float(part)
            if not (0 < fraction <= 1):
                raise ValueError(f"--top-token-fraction values must be in (0, 1], got {fraction}")
            if not any(np.isclose(fraction, existing) for existing in fractions):
                fractions.append(fraction)
    if not fractions:
        raise ValueError("At least one --top-token-fraction value is required.")
    return fractions


def top_token_fraction_suffix(fraction: float) -> str:
    percent = 100.0 * float(fraction)
    if np.isclose(percent, round(percent)):
        return f"top{int(round(percent)):02d}pct"
    compact = f"{float(fraction):.6g}".replace(".", "p")
    return f"frac{compact}"


def top_token_output_specs(fractions: Sequence[float]) -> List[Dict[str, Any]]:
    return [
        {
            "fraction": float(fraction),
            "suffix": top_token_fraction_suffix(fraction),
            "score_file": f"parcel_modality_attn_top_token_mean_{top_token_fraction_suffix(fraction)}.npy",
            "fraction_file": f"parcel_modality_attn_top_token_fraction_{top_token_fraction_suffix(fraction)}.npy",
        }
        for fraction in fractions
    ]


def primary_top_token_suffix(fractions: Sequence[float]) -> str:
    for fraction in fractions:
        if np.isclose(float(fraction), 0.25):
            return top_token_fraction_suffix(float(fraction))
    return top_token_fraction_suffix(float(fractions[0]))


def get_token_count(modality: str, backbone: str) -> int:
    backbone = backbone.lower()
    if modality == "video":
        if backbone not in VIDEO_TOKEN_COUNTS:
            raise ValueError(f"Unsupported video backbone: {backbone}")
        return VIDEO_TOKEN_COUNTS[backbone]
    if modality == "audio":
        if backbone not in AUDIO_TOKEN_COUNTS:
            raise ValueError(f"Unsupported audio backbone: {backbone}")
        return AUDIO_TOKEN_COUNTS[backbone]
    if modality == "text":
        if backbone not in TEXT_TOKEN_COUNTS:
            raise ValueError(f"Unsupported text backbone: {backbone}")
        return TEXT_TOKEN_COUNTS[backbone]
    raise ValueError(f"Unsupported modality: {modality}")


def build_modality_specs(args: argparse.Namespace) -> List[ModalitySpec]:
    backbones = {
        "video": args.video_backbone.lower(),
        "audio": args.audio_backbone.lower(),
        "text": args.text_backbone.lower(),
    }
    specs: List[ModalitySpec] = []
    start = 0
    for modality in normalize_modalities(args.modalities):
        token_count = get_token_count(modality, backbones[modality])
        specs.append(
            ModalitySpec(
                name=modality,
                backbone=backbones[modality],
                start=start,
                end=start + token_count,
            )
        )
        start += token_count
    return specs


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r") as f:
        return json.load(f)


def resolve_run_name(subject_id: int, run_name: str | None) -> str:
    if run_name is not None:
        return run_name
    if subject_id in DEFAULT_RUN_BY_SUBJECT:
        return DEFAULT_RUN_BY_SUBJECT[subject_id]
    raise ValueError(
        f"--run-name is required for subject_id={subject_id}; no default mapping is defined."
    )


def resolve_unit_paths(run_dir: Path, manifest: Dict[str, Any], explicit_units: Sequence[str] | None) -> List[Path]:
    if explicit_units:
        unit_names = list(explicit_units)
    elif manifest.get("units"):
        unit_names = [str(unit_info["unit"]) for unit_info in manifest["units"]]
    else:
        unit_names = sorted(path.stem for path in run_dir.glob("*.h5py"))

    paths = [run_dir / f"{unit_name}.h5py" for unit_name in unit_names]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing attention-map files: {missing}")
    return paths


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(json_ready(payload), f, indent=2)


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_token_metadata(path: Path, modality_specs: Sequence[ModalitySpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["token_index", "modality", "backbone", "modality_token_index"],
        )
        writer.writeheader()
        for spec in modality_specs:
            for token_index in range(spec.start, spec.end):
                writer.writerow(
                    {
                        "token_index": int(token_index),
                        "modality": spec.name,
                        "backbone": spec.backbone,
                        "modality_token_index": int(token_index - spec.start),
                    }
                )


def token_level_metadata(
    num_queries: int,
    num_memory_tokens: int,
    modality_specs: Sequence[ModalitySpec],
    enabled: bool = True,
) -> Dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "mean_shape": [int(num_queries), int(num_memory_tokens)],
            "fraction_shape": [int(num_queries), int(num_memory_tokens)],
            "modality_token_ranges": {
                spec.name: [int(spec.start), int(spec.end)] for spec in modality_specs
            },
            "note": "Token-level outputs were disabled for this run.",
        }
    return {
        "enabled": True,
        "mean_file": "parcel_token_attn_mean.npy",
        "fraction_file": "parcel_token_attn_fraction.npy",
        "metadata_file": "token_metadata.csv",
        "mean_shape": [int(num_queries), int(num_memory_tokens)],
        "fraction_shape": [int(num_queries), int(num_memory_tokens)],
        "modality_token_ranges": {
            spec.name: [int(spec.start), int(spec.end)] for spec in modality_specs
        },
        "note": (
            "Token-level attention preserves the memory-token axis and averages over samples "
            "and saved head slots. When attention_head_aggregation='mean', the saved head slot "
            "already represents the average over model heads."
        ),
    }


def row_fraction(values: np.ndarray) -> np.ndarray:
    return (
        values / np.clip(values.sum(axis=1, keepdims=True), 1e-12, None)
    ).astype(np.float32)


def validate_token_layout(
    num_memory_tokens: int,
    modality_specs: Sequence[ModalitySpec],
) -> None:
    expected = sum(spec.num_tokens for spec in modality_specs)
    if num_memory_tokens != expected:
        ranges = {spec.name: [spec.start, spec.end] for spec in modality_specs}
        raise ValueError(
            "Saved attention tensor token count does not match the configured modality backbones. "
            f"expected={expected}, found={num_memory_tokens}, ranges={ranges}"
        )


def build_query_to_parcel(num_queries: int) -> np.ndarray:
    return np.arange(1, num_queries + 1, dtype=np.int32)


def count_splits(split_values: np.ndarray) -> Dict[str, int]:
    if split_values.size == 0:
        return {}
    normalized = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in split_values]
    unique, counts = np.unique(normalized, return_counts=True)
    return {str(split): int(count) for split, count in zip(unique, counts)}


def normalize_split_array(split_values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in split_values],
        dtype=object,
    )


def chunk_bounds(num_samples: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, num_samples, chunk_size):
        yield start, min(start + chunk_size, num_samples)


def inspect_attention_dataset(
    attn_ds: h5py.Dataset,
    h5f: h5py.File,
) -> Dict[str, Any]:
    shape = tuple(int(v) for v in attn_ds.shape)
    if len(shape) == 4:
        saved_num_samples, stored_num_heads, num_queries, num_memory_tokens = shape
    elif len(shape) == 3:
        saved_num_samples, num_queries, num_memory_tokens = shape
        stored_num_heads = 1
    else:
        raise ValueError(
            f"Unsupported attention dataset rank {len(shape)} for {attn_ds.name}; "
            "expected [N, H, Q, T] or [N, Q, T]."
        )

    attention_head_aggregation = str(h5f.attrs.get("attention_head_aggregation", "unknown"))
    model_num_heads = int(h5f.attrs.get("model_num_heads", h5f.attrs.get("num_heads", stored_num_heads)))
    if attention_head_aggregation == "mean" and stored_num_heads != 1:
        raise ValueError(
            "Head-averaged attention should have exactly one stored head axis. "
            f"Found stored_num_heads={stored_num_heads} in {attn_ds.name}."
        )

    return {
        "dataset_shape": list(shape),
        "saved_num_samples": int(saved_num_samples),
        "stored_num_heads": int(stored_num_heads),
        "model_num_heads": int(model_num_heads),
        "num_queries": int(num_queries),
        "num_memory_tokens": int(num_memory_tokens),
        "attention_head_aggregation": attention_head_aggregation,
        "is_head_averaged": bool(attention_head_aggregation == "mean"),
    }


def read_attention_modality_slice(
    attn_ds: h5py.Dataset,
    start: int,
    end: int,
    spec: ModalitySpec,
) -> np.ndarray:
    if attn_ds.ndim == 4:
        return np.asarray(attn_ds[start:end, :, :, spec.start:spec.end], dtype=np.float32)
    if attn_ds.ndim == 3:
        return np.asarray(attn_ds[start:end, :, spec.start:spec.end], dtype=np.float32)[:, None, :, :]
    raise ValueError(
        f"Unsupported attention dataset rank {attn_ds.ndim} for {attn_ds.name}; expected 3 or 4."
    )


def top_fraction_mean(attn: np.ndarray, fraction: float) -> np.ndarray:
    if not (0 < fraction <= 1):
        raise ValueError(f"top-token fraction must be in (0, 1], got {fraction}")
    num_tokens = int(attn.shape[-1])
    k = max(1, int(np.ceil(fraction * num_tokens)))
    if k >= num_tokens:
        return attn.mean(axis=-1, dtype=np.float32)
    idx = np.argpartition(attn, -k, axis=-1)[..., -k:]
    return np.take_along_axis(attn, idx, axis=-1).mean(axis=-1, dtype=np.float32)


def build_time_windows(
    split_values: np.ndarray,
    ind_values: np.ndarray,
    window_size: int | None,
    window_stride: int,
) -> tuple[List[Dict[str, Any]], List[List[int]]]:
    if window_size is None:
        return [], [[] for _ in range(int(split_values.shape[0]))]
    if window_size <= 0:
        raise ValueError(f"--time-window-size must be positive, got {window_size}.")
    if window_stride <= 0:
        raise ValueError(f"--time-window-stride must be positive, got {window_stride}.")

    split_values = normalize_split_array(split_values)
    ind_values = np.asarray(ind_values, dtype=np.int32)
    sample_to_windows: List[List[int]] = [[] for _ in range(int(split_values.shape[0]))]
    windows: List[Dict[str, Any]] = []

    for split in sorted(set(split_values.tolist())):
        sample_indices = np.flatnonzero(split_values == split)
        if sample_indices.size == 0:
            continue
        order = np.argsort(ind_values[sample_indices], kind="stable")
        ordered_sample_indices = sample_indices[order]
        ordered_inds = ind_values[ordered_sample_indices]
        if ordered_sample_indices.size < window_size:
            continue

        local_window_index = 0
        for offset in range(0, ordered_sample_indices.size - window_size + 1, window_stride):
            member_indices = ordered_sample_indices[offset:offset + window_size]
            member_inds = ordered_inds[offset:offset + window_size]
            window_id = len(windows)
            for sample_idx in member_indices.tolist():
                sample_to_windows[int(sample_idx)].append(window_id)
            windows.append(
                {
                    "window_id": int(window_id),
                    "split": str(split),
                    "local_window_index": int(local_window_index),
                    "start_ind": int(member_inds[0]),
                    "end_ind": int(member_inds[-1]),
                    "num_samples": int(member_indices.size),
                    "sample_indices": [int(v) for v in member_indices.tolist()],
                    "sample_inds": [int(v) for v in member_inds.tolist()],
                }
            )
            local_window_index += 1

    return windows, sample_to_windows


def finalize_time_window_arrays(
    raw_sum: np.ndarray,
    norm_sum: np.ndarray,
    by_head_norm_sum: np.ndarray,
    window_counts: np.ndarray,
    stored_num_heads: int,
) -> Dict[str, np.ndarray]:
    if raw_sum.shape[0] == 0:
        num_queries = int(raw_sum.shape[1])
        num_modalities = int(raw_sum.shape[2])
        return {
            "parcel_modality_attn_raw_by_window": np.zeros((0, num_queries, num_modalities), dtype=np.float32),
            "parcel_modality_attn_norm_by_window": np.zeros((0, num_queries, num_modalities), dtype=np.float32),
            "parcel_modality_attn_by_saved_head_by_window": np.zeros(
                (0, stored_num_heads, num_queries, num_modalities), dtype=np.float32
            ),
        }

    safe_counts = np.maximum(window_counts.astype(np.float64), 1.0)
    raw = raw_sum / safe_counts[:, None, None] / float(stored_num_heads)
    norm = norm_sum / safe_counts[:, None, None] / float(stored_num_heads)
    by_head = by_head_norm_sum / safe_counts[:, None, None, None]
    return {
        "parcel_modality_attn_raw_by_window": raw.astype(np.float32),
        "parcel_modality_attn_norm_by_window": norm.astype(np.float32),
        "parcel_modality_attn_by_saved_head_by_window": by_head.astype(np.float32),
    }


def analyze_unit(
    h5_path: Path,
    modality_specs: Sequence[ModalitySpec],
    chunk_size: int,
    max_samples: int | None,
    qc_tol: float,
    time_window_size: int | None,
    time_window_stride: int,
    top_token_fractions: Sequence[float],
    save_token_level: bool,
    output_dir: Path | None = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as h5f:
        if "attn_maps" not in h5f:
            raise ValueError(f"{h5_path} does not contain an 'attn_maps' dataset.")
        attn_ds = h5f["attn_maps"]
        split_ds = h5f["split"]
        ind_ds = h5f["ind"]

        layout = inspect_attention_dataset(attn_ds, h5f)
        saved_num_samples = int(layout["saved_num_samples"])
        stored_num_heads = int(layout["stored_num_heads"])
        model_num_heads = int(layout["model_num_heads"])
        num_queries = int(layout["num_queries"])
        num_memory_tokens = int(layout["num_memory_tokens"])
        attention_head_aggregation = str(layout["attention_head_aggregation"])
        is_head_averaged = bool(layout["is_head_averaged"])
        num_samples = min(saved_num_samples, max_samples) if max_samples is not None else saved_num_samples
        if num_samples <= 0:
            raise ValueError(f"{h5_path} has no samples available for analysis.")
        validate_token_layout(num_memory_tokens, modality_specs)
        split_values = normalize_split_array(np.asarray(split_ds[:num_samples]))
        ind_values = np.asarray(ind_ds[:num_samples], dtype=np.int32)
        time_windows, sample_to_windows = build_time_windows(
            split_values=split_values,
            ind_values=ind_values,
            window_size=time_window_size,
            window_stride=time_window_stride,
        )

        raw_sum = np.zeros((num_queries, len(modality_specs)), dtype=np.float64)
        norm_sum = np.zeros((num_queries, len(modality_specs)), dtype=np.float64)
        token_sum = (
            np.zeros((num_queries, num_memory_tokens), dtype=np.float64)
            if save_token_level
            else None
        )
        top_token_specs = top_token_output_specs(top_token_fractions)
        top_token_mean_sums = {
            spec["suffix"]: np.zeros((num_queries, len(modality_specs)), dtype=np.float64)
            for spec in top_token_specs
        }
        per_head_norm_sum = np.zeros((stored_num_heads, num_queries, len(modality_specs)), dtype=np.float64)
        head_raw_sum = np.zeros((stored_num_heads, len(modality_specs)), dtype=np.float64)
        head_norm_sum = np.zeros((stored_num_heads, len(modality_specs)), dtype=np.float64)
        time_window_raw_sum = np.zeros((len(time_windows), num_queries, len(modality_specs)), dtype=np.float64)
        time_window_norm_sum = np.zeros((len(time_windows), num_queries, len(modality_specs)), dtype=np.float64)
        time_window_by_head_norm_sum = np.zeros(
            (len(time_windows), stored_num_heads, num_queries, len(modality_specs)),
            dtype=np.float64,
        )
        time_window_counts = np.zeros((len(time_windows),), dtype=np.int32)

        attn_sum_min = np.inf
        attn_sum_max = -np.inf
        attn_abs_err_sum = 0.0
        attn_abs_err_count = 0
        attn_abs_err_max = 0.0

        token_counts = np.asarray([spec.num_tokens for spec in modality_specs], dtype=np.float32)
        sample_head_denom = float(num_samples * stored_num_heads)
        sample_query_denom = float(num_samples * num_queries)

        chunk_iterator = chunk_bounds(num_samples, chunk_size)
        if show_progress:
            chunk_iterator = tqdm(
                chunk_iterator,
                total=(num_samples + chunk_size - 1) // chunk_size,
                desc=f"Analyze {h5_path.stem}",
                unit="chunk",
                leave=False,
            )

        for start, end in chunk_iterator:
            attn_sums = np.zeros((end - start, stored_num_heads, num_queries), dtype=np.float32)
            raw_chunk = np.zeros((end - start, stored_num_heads, num_queries, len(modality_specs)), dtype=np.float32)

            for modality_idx, spec in enumerate(modality_specs):
                modality_slice = read_attention_modality_slice(attn_ds, start, end, spec)
                if token_sum is not None:
                    token_sum[:, spec.start:spec.end] += modality_slice.sum(axis=(0, 1), dtype=np.float64)
                modality_sum = modality_slice.sum(axis=-1, dtype=np.float32)
                raw_chunk[..., modality_idx] = modality_sum
                for top_spec in top_token_specs:
                    top_token_mean_sums[top_spec["suffix"]][:, modality_idx] += top_fraction_mean(
                        modality_slice,
                        float(top_spec["fraction"]),
                    ).sum(axis=(0, 1), dtype=np.float64)
                attn_sums += modality_sum

            attn_sum_min = min(attn_sum_min, float(attn_sums.min()))
            attn_sum_max = max(attn_sum_max, float(attn_sums.max()))
            abs_err = np.abs(attn_sums - 1.0)
            attn_abs_err_sum += float(abs_err.sum(dtype=np.float64))
            attn_abs_err_count += int(abs_err.size)
            attn_abs_err_max = max(attn_abs_err_max, float(abs_err.max()))

            norm_chunk = raw_chunk / token_counts.reshape(1, 1, 1, -1)

            raw_sum += raw_chunk.sum(axis=(0, 1), dtype=np.float64)
            norm_sum += norm_chunk.sum(axis=(0, 1), dtype=np.float64)
            per_head_norm_sum += norm_chunk.sum(axis=0, dtype=np.float64)
            head_raw_sum += raw_chunk.sum(axis=(0, 2), dtype=np.float64)
            head_norm_sum += norm_chunk.sum(axis=(0, 2), dtype=np.float64)
            if len(time_windows) > 0:
                for offset in range(end - start):
                    sample_idx = start + offset
                    if not sample_to_windows[sample_idx]:
                        continue
                    sample_raw = raw_chunk[offset].sum(axis=0, dtype=np.float64)
                    sample_norm = norm_chunk[offset].sum(axis=0, dtype=np.float64)
                    sample_by_head_norm = norm_chunk[offset].astype(np.float64, copy=False)
                    for window_id in sample_to_windows[sample_idx]:
                        time_window_raw_sum[window_id] += sample_raw
                        time_window_norm_sum[window_id] += sample_norm
                        time_window_by_head_norm_sum[window_id] += sample_by_head_norm
                        time_window_counts[window_id] += 1

        parcel_modality_attn_raw = (raw_sum / sample_head_denom).astype(np.float32)
        parcel_modality_attn_norm = (norm_sum / sample_head_denom).astype(np.float32)
        if token_sum is not None:
            parcel_token_attn_mean = (token_sum / sample_head_denom).astype(np.float32)
            parcel_token_attn_fraction = row_fraction(parcel_token_attn_mean)
            token_fraction_row_sums = parcel_token_attn_fraction.sum(axis=1, dtype=np.float64)
            token_level_qc = {
                "token_fraction_row_sum_min": float(token_fraction_row_sums.min()),
                "token_fraction_row_sum_max": float(token_fraction_row_sums.max()),
                "token_fraction_row_sum_mean_abs_error": float(
                    np.abs(token_fraction_row_sums - 1.0).mean()
                ),
            }
        else:
            parcel_token_attn_mean = None
            parcel_token_attn_fraction = None
            token_level_qc = {}
        parcel_modality_attn_top_token_mean_by_fraction = {}
        parcel_modality_attn_top_token_fraction_by_fraction = {}
        for top_spec in top_token_specs:
            suffix = top_spec["suffix"]
            top_mean = (top_token_mean_sums[suffix] / sample_head_denom).astype(np.float32)
            parcel_modality_attn_top_token_mean_by_fraction[suffix] = top_mean
            parcel_modality_attn_top_token_fraction_by_fraction[suffix] = (
                top_mean / np.clip(top_mean.sum(axis=1, keepdims=True), 1e-12, None)
            ).astype(np.float32)
        primary_suffix = primary_top_token_suffix(top_token_fractions)
        parcel_modality_attn_top_token_mean = parcel_modality_attn_top_token_mean_by_fraction[primary_suffix]
        parcel_modality_attn_top_token_fraction = parcel_modality_attn_top_token_fraction_by_fraction[primary_suffix]
        parcel_modality_attn_per_head = (per_head_norm_sum / float(num_samples)).astype(np.float32)
        head_modality_attn_raw = (head_raw_sum / sample_query_denom).astype(np.float32)
        head_modality_attn_norm = (head_norm_sum / sample_query_denom).astype(np.float32)
        time_window_arrays = finalize_time_window_arrays(
            raw_sum=time_window_raw_sum,
            norm_sum=time_window_norm_sum,
            by_head_norm_sum=time_window_by_head_norm_sum,
            window_counts=time_window_counts,
            stored_num_heads=stored_num_heads,
        )
        query_to_parcel = build_query_to_parcel(num_queries)

        metadata = {
            "unit": h5_path.stem,
            "source_h5": str(h5_path),
            "num_samples_analyzed": int(num_samples),
            "num_samples_saved": int(saved_num_samples),
            "num_heads": int(stored_num_heads),
            "stored_num_heads": int(stored_num_heads),
            "model_num_heads": int(model_num_heads),
            "num_queries": int(num_queries),
            "num_memory_tokens": int(num_memory_tokens),
            "attention_head_aggregation": attention_head_aggregation,
            "is_head_averaged": is_head_averaged,
            "saved_head_axis_semantics": (
                "mean_over_model_heads" if is_head_averaged else "individual_model_heads"
            ),
            "modalities": [spec.name for spec in modality_specs],
            "modality_backbones": {spec.name: spec.backbone for spec in modality_specs},
            "modality_token_ranges": {
                spec.name: [int(spec.start), int(spec.end)] for spec in modality_specs
            },
            "token_count_normalization": {
                spec.name: int(spec.num_tokens) for spec in modality_specs
            },
            "token_level_outputs": token_level_metadata(
                num_queries=num_queries,
                num_memory_tokens=num_memory_tokens,
                modality_specs=modality_specs,
                enabled=save_token_level,
            ),
            "top_token_preference": {
                "enabled": True,
                "top_token_fractions": [float(spec["fraction"]) for spec in top_token_specs],
                "primary_top_token_fraction": float(
                    next(spec["fraction"] for spec in top_token_specs if spec["suffix"] == primary_suffix)
                ),
                "primary_suffix": primary_suffix,
                "score_file": "parcel_modality_attn_top_token_mean.npy",
                "fraction_file": "parcel_modality_attn_top_token_fraction.npy",
                "outputs": top_token_specs,
                "note": (
                    "For each parcel and modality, the score is the mean attention over the "
                    "top fraction of tokens within that modality, averaged over samples and stored head slots. "
                    "Fractions normalize those top-token scores across modalities within each parcel. "
                    "The unsuffixed score_file and fraction_file are aliases for primary_suffix."
                ),
            },
            "query_type": "parcel",
            "query_to_parcel_saved": True,
            "normalization_note": (
                "Normalized attention divides by static backbone token counts because valid-token masks "
                "are not saved by the exporter."
            ),
            "legacy_output_note": (
                "Files with 'per_head' or 'head_' in the name refer to stored head slots. "
                "When attention_head_aggregation='mean', there is a single stored slot that already "
                "represents the average over model heads."
            ),
            "time_window_config": {
                "enabled": bool(time_window_size is not None),
                "window_size": None if time_window_size is None else int(time_window_size),
                "window_stride": int(time_window_stride),
                "num_time_windows": int(len(time_windows)),
            },
        }
        quality_control = {
            "attn_maps_shape": list(layout["dataset_shape"]),
            "decoder_layers_saved": int(h5f.attrs.get("decoder_layers_saved", -1)),
            "stored_num_heads": int(stored_num_heads),
            "model_num_heads": int(model_num_heads),
            "attention_head_aggregation": attention_head_aggregation,
            "attn_sum_axis_last_min": attn_sum_min,
            "attn_sum_axis_last_max": attn_sum_max,
            "attn_sum_axis_last_mean_abs_error": (
                attn_abs_err_sum / max(attn_abs_err_count, 1)
            ),
            "attn_sum_axis_last_max_abs_error": attn_abs_err_max,
            "attn_sum_axis_last_within_tolerance": bool(attn_abs_err_max <= qc_tol),
            "qc_tolerance": qc_tol,
            **token_level_qc,
            "split_counts_analyzed": count_splits(split_values),
            "time_window_config": {
                "enabled": bool(time_window_size is not None),
                "window_size": None if time_window_size is None else int(time_window_size),
                "window_stride": int(time_window_stride),
                "num_time_windows": int(len(time_windows)),
            },
        }

        if output_dir is not None:
            save_array(output_dir / "parcel_modality_attn_raw.npy", parcel_modality_attn_raw)
            save_array(output_dir / "parcel_modality_attn_norm.npy", parcel_modality_attn_norm)
            if save_token_level:
                save_array(output_dir / "parcel_token_attn_mean.npy", parcel_token_attn_mean)
                save_array(output_dir / "parcel_token_attn_fraction.npy", parcel_token_attn_fraction)
                save_token_metadata(output_dir / "token_metadata.csv", modality_specs)
            for top_spec in top_token_specs:
                suffix = top_spec["suffix"]
                save_array(
                    output_dir / str(top_spec["score_file"]),
                    parcel_modality_attn_top_token_mean_by_fraction[suffix],
                )
                save_array(
                    output_dir / str(top_spec["fraction_file"]),
                    parcel_modality_attn_top_token_fraction_by_fraction[suffix],
                )
            save_array(output_dir / "parcel_modality_attn_top_token_mean.npy", parcel_modality_attn_top_token_mean)
            save_array(
                output_dir / "parcel_modality_attn_top_token_fraction.npy",
                parcel_modality_attn_top_token_fraction,
            )
            save_array(output_dir / "parcel_modality_attn_per_head.npy", parcel_modality_attn_per_head)
            save_array(output_dir / "head_modality_attn_raw.npy", head_modality_attn_raw)
            save_array(output_dir / "head_modality_attn_norm.npy", head_modality_attn_norm)
            save_array(output_dir / "parcel_modality_attn_by_saved_head.npy", parcel_modality_attn_per_head)
            save_array(output_dir / "saved_head_modality_attn_raw.npy", head_modality_attn_raw)
            save_array(output_dir / "saved_head_modality_attn_norm.npy", head_modality_attn_norm)
            if is_head_averaged:
                save_array(
                    output_dir / "parcel_modality_attn_head_aggregated.npy",
                    np.squeeze(parcel_modality_attn_per_head, axis=0),
                )
                save_array(
                    output_dir / "modality_attn_head_aggregated_raw.npy",
                    np.squeeze(head_modality_attn_raw, axis=0),
                )
                save_array(
                    output_dir / "modality_attn_head_aggregated_norm.npy",
                    np.squeeze(head_modality_attn_norm, axis=0),
                )
            if time_window_size is not None:
                save_array(
                    output_dir / "parcel_modality_attn_raw_by_window.npy",
                    time_window_arrays["parcel_modality_attn_raw_by_window"],
                )
                save_array(
                    output_dir / "parcel_modality_attn_norm_by_window.npy",
                    time_window_arrays["parcel_modality_attn_norm_by_window"],
                )
                save_array(
                    output_dir / "parcel_modality_attn_by_saved_head_by_window.npy",
                    time_window_arrays["parcel_modality_attn_by_saved_head_by_window"],
                )
                if is_head_averaged:
                    save_array(
                        output_dir / "parcel_modality_attn_head_aggregated_by_window.npy",
                        np.squeeze(time_window_arrays["parcel_modality_attn_by_saved_head_by_window"], axis=1),
                    )
                save_json(output_dir / "time_window_metadata.json", {"windows": time_windows})
            save_array(output_dir / "query_to_parcel.npy", query_to_parcel)
            save_json(output_dir / "metadata.json", metadata)
            save_json(output_dir / "quality_control.json", quality_control)

    return {
        "unit": h5_path.stem,
        "num_samples_analyzed": int(num_samples),
        "num_samples_saved": int(saved_num_samples),
        "num_heads": int(stored_num_heads),
        "stored_num_heads": int(stored_num_heads),
        "model_num_heads": int(model_num_heads),
        "num_queries": int(num_queries),
        "num_memory_tokens": int(num_memory_tokens),
        "attention_head_aggregation": attention_head_aggregation,
        "is_head_averaged": is_head_averaged,
        "save_token_level": bool(save_token_level),
        "raw_sum": raw_sum,
        "norm_sum": norm_sum,
        "token_sum": token_sum,
        "top_token_specs": top_token_specs,
        "top_token_mean_sums": top_token_mean_sums,
        "per_head_norm_sum": per_head_norm_sum,
        "head_raw_sum": head_raw_sum,
        "head_norm_sum": head_norm_sum,
        "query_to_parcel": query_to_parcel,
        "time_windows": time_windows,
        "time_window_counts": time_window_counts,
        "time_window_arrays": time_window_arrays,
        "metadata": metadata,
        "quality_control": quality_control,
    }


def save_combined_outputs(
    output_dir: Path,
    combined_result: Dict[str, Any],
    modality_specs: Sequence[ModalitySpec],
    run_dir: Path,
    manifest: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    combined_metadata = {
        "subject_id": int(args.subject_id),
        "run_name": str(run_dir.name),
        "source_run_dir": str(run_dir),
        "modalities": [spec.name for spec in modality_specs],
        "modality_backbones": {spec.name: spec.backbone for spec in modality_specs},
        "modality_token_ranges": {
            spec.name: [int(spec.start), int(spec.end)] for spec in modality_specs
        },
        "num_units": int(combined_result["num_units"]),
        "unit_names": combined_result["unit_names"],
        "num_queries": int(combined_result["num_queries"]),
        "num_memory_tokens": int(combined_result["num_memory_tokens"]),
        "num_heads": int(combined_result["num_heads"]),
        "stored_num_heads": int(combined_result["num_heads"]),
        "model_num_heads": int(combined_result["model_num_heads"]),
        "attention_head_aggregation": combined_result["attention_head_aggregation"],
        "is_head_averaged": bool(combined_result["is_head_averaged"]),
        "combined_weighting": "implicit_sample_weighting_via_streamed_global_sums",
        "manifest_checkpoint": manifest.get("checkpoint"),
        "manifest_test_splits": manifest.get("test_splits"),
        "normalization_note": (
            "Normalized attention divides by static backbone token counts because valid-token masks "
            "are not saved by the exporter."
        ),
        "token_level_outputs": token_level_metadata(
            num_queries=int(combined_result["num_queries"]),
            num_memory_tokens=int(combined_result["num_memory_tokens"]),
            modality_specs=modality_specs,
            enabled=bool(combined_result["save_token_level"]),
        ),
        "top_token_preference": {
            "enabled": True,
            "top_token_fractions": [float(spec["fraction"]) for spec in combined_result["top_token_specs"]],
            "primary_top_token_fraction": float(combined_result["primary_top_token_fraction"]),
            "primary_suffix": combined_result["primary_top_token_suffix"],
            "score_file": "parcel_modality_attn_top_token_mean.npy",
            "fraction_file": "parcel_modality_attn_top_token_fraction.npy",
            "outputs": combined_result["top_token_specs"],
            "note": (
                "For each parcel and modality, the score is the mean attention over the top fraction "
                "of tokens within that modality, averaged over samples, unit shards, and stored head slots. "
                "Fractions normalize those top-token scores across modalities within each parcel. "
                "The unsuffixed score_file and fraction_file are aliases for primary_suffix."
            ),
        },
        "legacy_output_note": (
            "Files with 'per_head' or 'head_' in the name refer to stored head slots. "
            "When attention_head_aggregation='mean', there is a single stored slot that already "
            "represents the average over model heads."
        ),
        "time_window_config": combined_result["time_window_config"],
    }
    quality_control = {
        "subject_id": int(args.subject_id),
        "run_name": str(run_dir.name),
        "source_run_dir": str(run_dir),
        "chunk_size": int(args.chunk_size),
        "max_samples": None if args.max_samples is None else int(args.max_samples),
        "num_units": int(combined_result["num_units"]),
        "num_samples_analyzed": int(combined_result["num_samples_analyzed"]),
        "unit_summaries": combined_result["unit_summaries"],
        "stored_num_heads": int(combined_result["num_heads"]),
        "model_num_heads": int(combined_result["model_num_heads"]),
        "attention_head_aggregation": combined_result["attention_head_aggregation"],
        "time_window_config": combined_result["time_window_config"],
        "attn_sum_axis_last_min": combined_result["attn_sum_min"],
        "attn_sum_axis_last_max": combined_result["attn_sum_max"],
        "attn_sum_axis_last_mean_abs_error": combined_result["attn_abs_err_mean"],
        "attn_sum_axis_last_max_abs_error": combined_result["attn_abs_err_max"],
        "attn_sum_axis_last_within_tolerance": bool(combined_result["attn_abs_err_max"] <= args.qc_tol),
        "qc_tolerance": float(args.qc_tol),
    }
    if combined_result["save_token_level"]:
        quality_control.update(
            {
                "token_fraction_row_sum_min": combined_result["token_fraction_row_sum_min"],
                "token_fraction_row_sum_max": combined_result["token_fraction_row_sum_max"],
                "token_fraction_row_sum_mean_abs_error": combined_result[
                    "token_fraction_row_sum_mean_abs_error"
                ],
            }
        )

    save_array(output_dir / "parcel_modality_attn_raw.npy", combined_result["parcel_modality_attn_raw"])
    save_array(output_dir / "parcel_modality_attn_norm.npy", combined_result["parcel_modality_attn_norm"])
    if combined_result["save_token_level"]:
        save_array(output_dir / "parcel_token_attn_mean.npy", combined_result["parcel_token_attn_mean"])
        save_array(output_dir / "parcel_token_attn_fraction.npy", combined_result["parcel_token_attn_fraction"])
        save_token_metadata(output_dir / "token_metadata.csv", modality_specs)
    for top_spec in combined_result["top_token_specs"]:
        suffix = top_spec["suffix"]
        save_array(
            output_dir / str(top_spec["score_file"]),
            combined_result["parcel_modality_attn_top_token_mean_by_fraction"][suffix],
        )
        save_array(
            output_dir / str(top_spec["fraction_file"]),
            combined_result["parcel_modality_attn_top_token_fraction_by_fraction"][suffix],
        )
    save_array(output_dir / "parcel_modality_attn_top_token_mean.npy", combined_result["parcel_modality_attn_top_token_mean"])
    save_array(
        output_dir / "parcel_modality_attn_top_token_fraction.npy",
        combined_result["parcel_modality_attn_top_token_fraction"],
    )
    save_array(output_dir / "parcel_modality_attn_per_head.npy", combined_result["parcel_modality_attn_per_head"])
    save_array(output_dir / "head_modality_attn_raw.npy", combined_result["head_modality_attn_raw"])
    save_array(output_dir / "head_modality_attn_norm.npy", combined_result["head_modality_attn_norm"])
    save_array(output_dir / "parcel_modality_attn_by_saved_head.npy", combined_result["parcel_modality_attn_per_head"])
    save_array(output_dir / "saved_head_modality_attn_raw.npy", combined_result["head_modality_attn_raw"])
    save_array(output_dir / "saved_head_modality_attn_norm.npy", combined_result["head_modality_attn_norm"])
    if combined_result["is_head_averaged"]:
        save_array(
            output_dir / "parcel_modality_attn_head_aggregated.npy",
            np.squeeze(combined_result["parcel_modality_attn_per_head"], axis=0),
        )
        save_array(
            output_dir / "modality_attn_head_aggregated_raw.npy",
            np.squeeze(combined_result["head_modality_attn_raw"], axis=0),
        )
        save_array(
            output_dir / "modality_attn_head_aggregated_norm.npy",
            np.squeeze(combined_result["head_modality_attn_norm"], axis=0),
        )
    if combined_result["time_window_config"]["enabled"]:
        save_array(
            output_dir / "parcel_modality_attn_raw_by_window.npy",
            combined_result["parcel_modality_attn_raw_by_window"],
        )
        save_array(
            output_dir / "parcel_modality_attn_norm_by_window.npy",
            combined_result["parcel_modality_attn_norm_by_window"],
        )
        save_array(
            output_dir / "parcel_modality_attn_by_saved_head_by_window.npy",
            combined_result["parcel_modality_attn_by_saved_head_by_window"],
        )
        if combined_result["is_head_averaged"]:
            save_array(
                output_dir / "parcel_modality_attn_head_aggregated_by_window.npy",
                np.squeeze(combined_result["parcel_modality_attn_by_saved_head_by_window"], axis=1),
            )
        save_json(output_dir / "time_window_metadata.json", {"windows": combined_result["time_windows"]})
    save_array(output_dir / "query_to_parcel.npy", combined_result["query_to_parcel"])
    save_json(output_dir / "metadata.json", combined_metadata)
    save_json(output_dir / "quality_control.json", quality_control)


def accumulate_combined_results(unit_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not unit_results:
        raise ValueError("No unit results to combine.")

    first = unit_results[0]
    num_queries = int(first["num_queries"])
    num_memory_tokens = int(first["num_memory_tokens"])
    num_heads = int(first["num_heads"])
    model_num_heads = int(first["model_num_heads"])
    num_modalities = int(first["raw_sum"].shape[1])
    attention_head_aggregation = str(first["attention_head_aggregation"])
    is_head_averaged = bool(first["is_head_averaged"])
    save_token_level = bool(first["save_token_level"])
    modality_token_ranges = first["metadata"]["modality_token_ranges"]
    top_token_specs = first["top_token_specs"]
    top_token_suffixes = [str(spec["suffix"]) for spec in top_token_specs]
    time_window_enabled = bool(first["metadata"]["time_window_config"]["enabled"])
    time_window_size = first["metadata"]["time_window_config"]["window_size"]
    time_window_stride = int(first["metadata"]["time_window_config"]["window_stride"])

    total_samples = float(sum(result["num_samples_analyzed"] for result in unit_results))
    if total_samples <= 0:
        raise ValueError("No analyzed samples available for combined outputs.")

    total_raw_sum = np.zeros((num_queries, num_modalities), dtype=np.float64)
    total_norm_sum = np.zeros((num_queries, num_modalities), dtype=np.float64)
    total_token_sum = (
        np.zeros((num_queries, num_memory_tokens), dtype=np.float64)
        if save_token_level
        else None
    )
    total_top_token_mean_sums = {
        suffix: np.zeros((num_queries, num_modalities), dtype=np.float64)
        for suffix in top_token_suffixes
    }
    total_per_head_norm_sum = np.zeros((num_heads, num_queries, num_modalities), dtype=np.float64)
    total_head_raw_sum = np.zeros((num_heads, num_modalities), dtype=np.float64)
    total_head_norm_sum = np.zeros((num_heads, num_modalities), dtype=np.float64)

    attn_sum_min = np.inf
    attn_sum_max = -np.inf
    attn_abs_err_weighted_sum = 0.0
    attn_abs_err_count = 0
    attn_abs_err_max = 0.0
    unit_summaries = []
    combined_time_windows: List[Dict[str, Any]] = []
    combined_time_window_raw: List[np.ndarray] = []
    combined_time_window_norm: List[np.ndarray] = []
    combined_time_window_by_head: List[np.ndarray] = []

    for result in unit_results:
        if int(result["num_queries"]) != num_queries:
            raise ValueError("Query count differs across unit files.")
        if int(result["num_memory_tokens"]) != num_memory_tokens:
            raise ValueError("Memory-token count differs across unit files.")
        if int(result["num_heads"]) != num_heads:
            raise ValueError("Stored head count differs across unit files.")
        if int(result["model_num_heads"]) != model_num_heads:
            raise ValueError("Model head count differs across unit files.")
        if str(result["attention_head_aggregation"]) != attention_head_aggregation:
            raise ValueError("Attention head aggregation mode differs across unit files.")
        if bool(result["save_token_level"]) != save_token_level:
            raise ValueError("Token-level output configuration differs across unit files.")
        if result["metadata"]["modality_token_ranges"] != modality_token_ranges:
            raise ValueError("Modality token ranges differ across unit files.")
        result_top_token_suffixes = [str(spec["suffix"]) for spec in result["top_token_specs"]]
        if result_top_token_suffixes != top_token_suffixes:
            raise ValueError("Top-token fractions differ across unit files.")
        result_tw_cfg = result["metadata"]["time_window_config"]
        if bool(result_tw_cfg["enabled"]) != time_window_enabled:
            raise ValueError("Time-window configuration differs across unit files.")
        if result_tw_cfg["window_size"] != time_window_size:
            raise ValueError("Time-window size differs across unit files.")
        if int(result_tw_cfg["window_stride"]) != time_window_stride:
            raise ValueError("Time-window stride differs across unit files.")
        total_raw_sum += result["raw_sum"].astype(np.float64)
        total_norm_sum += result["norm_sum"].astype(np.float64)
        if total_token_sum is not None:
            total_token_sum += result["token_sum"].astype(np.float64)
        for suffix in top_token_suffixes:
            total_top_token_mean_sums[suffix] += result["top_token_mean_sums"][suffix].astype(np.float64)
        total_per_head_norm_sum += result["per_head_norm_sum"].astype(np.float64)
        total_head_raw_sum += result["head_raw_sum"].astype(np.float64)
        total_head_norm_sum += result["head_norm_sum"].astype(np.float64)

        qc = result["quality_control"]
        attn_sum_min = min(attn_sum_min, float(qc["attn_sum_axis_last_min"]))
        attn_sum_max = max(attn_sum_max, float(qc["attn_sum_axis_last_max"]))
        attn_abs_err_max = max(attn_abs_err_max, float(qc["attn_sum_axis_last_max_abs_error"]))

        count = int(result["num_samples_analyzed"]) * int(result["num_heads"]) * int(result["num_queries"])
        attn_abs_err_count += count
        attn_abs_err_weighted_sum += float(qc["attn_sum_axis_last_mean_abs_error"]) * count
        if time_window_enabled:
            combined_time_window_raw.append(result["time_window_arrays"]["parcel_modality_attn_raw_by_window"])
            combined_time_window_norm.append(result["time_window_arrays"]["parcel_modality_attn_norm_by_window"])
            combined_time_window_by_head.append(
                result["time_window_arrays"]["parcel_modality_attn_by_saved_head_by_window"]
            )
            for window in result["time_windows"]:
                global_window_id = len(combined_time_windows)
                combined_time_windows.append(
                    {
                        "global_window_id": int(global_window_id),
                        "unit": result["unit"],
                        **window,
                    }
                )
        unit_summaries.append(
            {
                "unit": result["unit"],
                "num_samples_analyzed": int(result["num_samples_analyzed"]),
                "num_samples_saved": int(result["num_samples_saved"]),
                "num_heads": int(result["num_heads"]),
                "stored_num_heads": int(result["stored_num_heads"]),
                "model_num_heads": int(result["model_num_heads"]),
                "num_queries": int(result["num_queries"]),
                "num_memory_tokens": int(result["num_memory_tokens"]),
                "attention_head_aggregation": str(result["attention_head_aggregation"]),
                "attn_maps_shape": result["quality_control"]["attn_maps_shape"],
                "num_time_windows": int(len(result["time_windows"])),
            }
        )

    if time_window_enabled:
        if combined_time_window_raw:
            parcel_modality_attn_raw_by_window = np.concatenate(combined_time_window_raw, axis=0).astype(np.float32)
            parcel_modality_attn_norm_by_window = np.concatenate(combined_time_window_norm, axis=0).astype(np.float32)
            parcel_modality_attn_by_saved_head_by_window = np.concatenate(
                combined_time_window_by_head, axis=0
            ).astype(np.float32)
        else:
            parcel_modality_attn_raw_by_window = np.zeros((0, num_queries, num_modalities), dtype=np.float32)
            parcel_modality_attn_norm_by_window = np.zeros((0, num_queries, num_modalities), dtype=np.float32)
            parcel_modality_attn_by_saved_head_by_window = np.zeros(
                (0, num_heads, num_queries, num_modalities), dtype=np.float32
            )
    else:
        parcel_modality_attn_raw_by_window = np.zeros((0, num_queries, num_modalities), dtype=np.float32)
        parcel_modality_attn_norm_by_window = np.zeros((0, num_queries, num_modalities), dtype=np.float32)
        parcel_modality_attn_by_saved_head_by_window = np.zeros(
            (0, num_heads, num_queries, num_modalities), dtype=np.float32
        )

    parcel_modality_attn_top_token_mean_by_fraction = {}
    parcel_modality_attn_top_token_fraction_by_fraction = {}
    for suffix in top_token_suffixes:
        top_mean = (total_top_token_mean_sums[suffix] / float(num_heads * total_samples)).astype(np.float32)
        parcel_modality_attn_top_token_mean_by_fraction[suffix] = top_mean
        parcel_modality_attn_top_token_fraction_by_fraction[suffix] = (
            top_mean / np.clip(top_mean.sum(axis=1, keepdims=True), 1e-12, None)
        ).astype(np.float32)
    primary_suffix = primary_top_token_suffix([float(spec["fraction"]) for spec in top_token_specs])
    primary_fraction = next(float(spec["fraction"]) for spec in top_token_specs if spec["suffix"] == primary_suffix)
    parcel_modality_attn_top_token_mean = parcel_modality_attn_top_token_mean_by_fraction[primary_suffix]
    parcel_modality_attn_top_token_fraction = parcel_modality_attn_top_token_fraction_by_fraction[primary_suffix]
    if total_token_sum is not None:
        parcel_token_attn_mean = (total_token_sum / float(num_heads * total_samples)).astype(np.float32)
        parcel_token_attn_fraction = row_fraction(parcel_token_attn_mean)
        token_fraction_row_sums = parcel_token_attn_fraction.sum(axis=1, dtype=np.float64)
        token_level_qc = {
            "token_fraction_row_sum_min": float(token_fraction_row_sums.min()),
            "token_fraction_row_sum_max": float(token_fraction_row_sums.max()),
            "token_fraction_row_sum_mean_abs_error": float(
                np.abs(token_fraction_row_sums - 1.0).mean()
            ),
        }
    else:
        parcel_token_attn_mean = None
        parcel_token_attn_fraction = None
        token_level_qc = {}

    return {
        "num_units": len(unit_results),
        "unit_names": [result["unit"] for result in unit_results],
        "num_samples_analyzed": int(total_samples),
        "num_queries": num_queries,
        "num_memory_tokens": num_memory_tokens,
        "num_heads": num_heads,
        "model_num_heads": model_num_heads,
        "attention_head_aggregation": attention_head_aggregation,
        "is_head_averaged": is_head_averaged,
        "save_token_level": save_token_level,
        "top_token_specs": top_token_specs,
        "primary_top_token_suffix": primary_suffix,
        "primary_top_token_fraction": primary_fraction,
        "time_window_config": {
            "enabled": time_window_enabled,
            "window_size": time_window_size,
            "window_stride": time_window_stride,
            "num_time_windows": int(len(combined_time_windows)),
        },
        "parcel_modality_attn_raw": (total_raw_sum / float(num_heads * total_samples)).astype(np.float32),
        "parcel_modality_attn_norm": (total_norm_sum / float(num_heads * total_samples)).astype(np.float32),
        "parcel_token_attn_mean": parcel_token_attn_mean,
        "parcel_token_attn_fraction": parcel_token_attn_fraction,
        "parcel_modality_attn_top_token_mean": parcel_modality_attn_top_token_mean,
        "parcel_modality_attn_top_token_fraction": parcel_modality_attn_top_token_fraction,
        "parcel_modality_attn_top_token_mean_by_fraction": parcel_modality_attn_top_token_mean_by_fraction,
        "parcel_modality_attn_top_token_fraction_by_fraction": parcel_modality_attn_top_token_fraction_by_fraction,
        "parcel_modality_attn_per_head": (total_per_head_norm_sum / total_samples).astype(np.float32),
        "head_modality_attn_raw": (total_head_raw_sum / float(num_queries * total_samples)).astype(np.float32),
        "head_modality_attn_norm": (total_head_norm_sum / float(num_queries * total_samples)).astype(np.float32),
        "parcel_modality_attn_raw_by_window": parcel_modality_attn_raw_by_window,
        "parcel_modality_attn_norm_by_window": parcel_modality_attn_norm_by_window,
        "parcel_modality_attn_by_saved_head_by_window": parcel_modality_attn_by_saved_head_by_window,
        "time_windows": combined_time_windows,
        "query_to_parcel": build_query_to_parcel(num_queries),
        "attn_sum_min": attn_sum_min,
        "attn_sum_max": attn_sum_max,
        "attn_abs_err_mean": attn_abs_err_weighted_sum / max(attn_abs_err_count, 1),
        "attn_abs_err_max": attn_abs_err_max,
        **token_level_qc,
        "unit_summaries": unit_summaries,
    }


def main() -> None:
    args = build_parser().parse_args()
    args.modalities = normalize_modalities(args.modalities)
    args.top_token_fraction = normalize_top_token_fractions(args.top_token_fraction)
    args.run_name = resolve_run_name(args.subject_id, args.run_name)

    run_dir = args.attn_root / str(args.subject_id) / args.run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Attention-map run directory not found: {run_dir}")

    manifest = load_manifest(run_dir)
    modality_specs = build_modality_specs(args)
    unit_paths = resolve_unit_paths(run_dir, manifest, args.units)

    run_save_dir = args.save_root / str(args.subject_id) / args.run_name
    unit_results = []
    unit_iterator = tqdm(unit_paths, desc="Units", unit="unit")
    for unit_path in unit_iterator:
        output_dir = (run_save_dir / unit_path.stem) if args.save_per_unit else None
        result = analyze_unit(
            h5_path=unit_path,
            modality_specs=modality_specs,
            chunk_size=args.chunk_size,
            max_samples=args.max_samples,
            qc_tol=args.qc_tol,
            time_window_size=args.time_window_size,
            time_window_stride=args.time_window_stride,
            top_token_fractions=args.top_token_fraction,
            save_token_level=args.save_token_level,
            output_dir=output_dir,
            show_progress=True,
        )
        unit_results.append(result)
        unit_iterator.set_postfix(
            unit=unit_path.stem,
            samples=result["num_samples_analyzed"],
            heads=result["num_heads"],
        )
        tqdm.write(
            f"[unit={unit_path.stem}] samples={result['num_samples_analyzed']} "
            f"queries={result['num_queries']} heads={result['num_heads']} "
            f"per_unit_saved={bool(output_dir)}"
        )

    combined_result = accumulate_combined_results(unit_results)
    save_combined_outputs(
        output_dir=run_save_dir,
        combined_result=combined_result,
        modality_specs=modality_specs,
        run_dir=run_dir,
        manifest=manifest,
        args=args,
    )
    print(f"[combined] saved_to={run_save_dir}")



if __name__ == "__main__":
    main()
