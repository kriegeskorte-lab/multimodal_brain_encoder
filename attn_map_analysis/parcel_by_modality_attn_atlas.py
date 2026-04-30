from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import h5py
import numpy as np
from tqdm.auto import tqdm

'''
pixi run python ./attn_map_analysis/parcel_by_modality_attn_atlas.py \
  --subject-id 2 \
  --video-backbone dino \
  --audio-backbone whisper \
  --text-backbone llama
'''


DEFAULT_RUN_BY_SUBJECT = {
    1: "04-07-2026-16-09", # parcel
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
    return parser


def normalize_modalities(modalities: Sequence[str]) -> List[str]:
    normalized = [str(modality).strip().lower() for modality in modalities if str(modality).strip()]
    invalid = [modality for modality in normalized if modality not in MODALITY_ORDER]
    if invalid:
        raise ValueError(f"Unsupported modalities: {invalid}")
    return [modality for modality in MODALITY_ORDER if modality in normalized]


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


def chunk_bounds(num_samples: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, num_samples, chunk_size):
        yield start, min(start + chunk_size, num_samples)


def analyze_unit(
    h5_path: Path,
    modality_specs: Sequence[ModalitySpec],
    chunk_size: int,
    max_samples: int | None,
    qc_tol: float,
    output_dir: Path | None = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as h5f:
        if "attn_maps" not in h5f:
            raise ValueError(f"{h5_path} does not contain an 'attn_maps' dataset.")
        attn_ds = h5f["attn_maps"]
        split_ds = h5f["split"]

        saved_num_samples, num_heads, num_queries, num_memory_tokens = map(int, attn_ds.shape)
        num_samples = min(saved_num_samples, max_samples) if max_samples is not None else saved_num_samples
        if num_samples <= 0:
            raise ValueError(f"{h5_path} has no samples available for analysis.")
        validate_token_layout(num_memory_tokens, modality_specs)

        raw_sum = np.zeros((num_queries, len(modality_specs)), dtype=np.float64)
        norm_sum = np.zeros((num_queries, len(modality_specs)), dtype=np.float64)
        per_head_norm_sum = np.zeros((num_heads, num_queries, len(modality_specs)), dtype=np.float64)
        head_raw_sum = np.zeros((num_heads, len(modality_specs)), dtype=np.float64)
        head_norm_sum = np.zeros((num_heads, len(modality_specs)), dtype=np.float64)

        attn_sum_min = np.inf
        attn_sum_max = -np.inf
        attn_abs_err_sum = 0.0
        attn_abs_err_count = 0
        attn_abs_err_max = 0.0

        token_counts = np.asarray([spec.num_tokens for spec in modality_specs], dtype=np.float32)
        sample_head_denom = float(num_samples * num_heads)
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
            attn_sums = np.zeros((end - start, num_heads, num_queries), dtype=np.float32)
            raw_chunk = np.zeros((end - start, num_heads, num_queries, len(modality_specs)), dtype=np.float32)

            for modality_idx, spec in enumerate(modality_specs):
                modality_slice = attn_ds[start:end, :, :, spec.start:spec.end]
                modality_sum = modality_slice.sum(axis=-1, dtype=np.float32)
                raw_chunk[..., modality_idx] = modality_sum
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

        parcel_modality_attn_raw = (raw_sum / sample_head_denom).astype(np.float32)
        parcel_modality_attn_norm = (norm_sum / sample_head_denom).astype(np.float32)
        parcel_modality_attn_per_head = (per_head_norm_sum / float(num_samples)).astype(np.float32)
        head_modality_attn_raw = (head_raw_sum / sample_query_denom).astype(np.float32)
        head_modality_attn_norm = (head_norm_sum / sample_query_denom).astype(np.float32)
        query_to_parcel = build_query_to_parcel(num_queries)

        metadata = {
            "unit": h5_path.stem,
            "source_h5": str(h5_path),
            "num_samples_analyzed": int(num_samples),
            "num_samples_saved": int(saved_num_samples),
            "num_heads": int(num_heads),
            "num_queries": int(num_queries),
            "num_memory_tokens": int(num_memory_tokens),
            "modalities": [spec.name for spec in modality_specs],
            "modality_backbones": {spec.name: spec.backbone for spec in modality_specs},
            "modality_token_ranges": {
                spec.name: [int(spec.start), int(spec.end)] for spec in modality_specs
            },
            "token_count_normalization": {
                spec.name: int(spec.num_tokens) for spec in modality_specs
            },
            "query_type": "parcel",
            "query_to_parcel_saved": True,
            "normalization_note": (
                "Normalized attention divides by static backbone token counts because valid-token masks "
                "are not saved by the exporter."
            ),
        }
        quality_control = {
            "attn_maps_shape": [int(v) for v in attn_ds.shape],
            "decoder_layers_saved": int(h5f.attrs.get("decoder_layers_saved", -1)),
            "attn_sum_axis_last_min": attn_sum_min,
            "attn_sum_axis_last_max": attn_sum_max,
            "attn_sum_axis_last_mean_abs_error": (
                attn_abs_err_sum / max(attn_abs_err_count, 1)
            ),
            "attn_sum_axis_last_max_abs_error": attn_abs_err_max,
            "attn_sum_axis_last_within_tolerance": bool(attn_abs_err_max <= qc_tol),
            "qc_tolerance": qc_tol,
            "split_counts_analyzed": count_splits(np.asarray(split_ds[:num_samples])),
        }

        if output_dir is not None:
            save_array(output_dir / "parcel_modality_attn_raw.npy", parcel_modality_attn_raw)
            save_array(output_dir / "parcel_modality_attn_norm.npy", parcel_modality_attn_norm)
            save_array(output_dir / "parcel_modality_attn_per_head.npy", parcel_modality_attn_per_head)
            save_array(output_dir / "head_modality_attn_raw.npy", head_modality_attn_raw)
            save_array(output_dir / "head_modality_attn_norm.npy", head_modality_attn_norm)
            save_array(output_dir / "query_to_parcel.npy", query_to_parcel)
            save_json(output_dir / "metadata.json", metadata)
            save_json(output_dir / "quality_control.json", quality_control)

    return {
        "unit": h5_path.stem,
        "num_samples_analyzed": int(num_samples),
        "num_samples_saved": int(saved_num_samples),
        "num_heads": int(num_heads),
        "num_queries": int(num_queries),
        "num_memory_tokens": int(num_memory_tokens),
        "raw_sum": raw_sum,
        "norm_sum": norm_sum,
        "per_head_norm_sum": per_head_norm_sum,
        "head_raw_sum": head_raw_sum,
        "head_norm_sum": head_norm_sum,
        "query_to_parcel": query_to_parcel,
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
        "num_heads": int(combined_result["num_heads"]),
        "combined_weighting": "implicit_sample_weighting_via_streamed_global_sums",
        "manifest_checkpoint": manifest.get("checkpoint"),
        "manifest_test_splits": manifest.get("test_splits"),
        "normalization_note": (
            "Normalized attention divides by static backbone token counts because valid-token masks "
            "are not saved by the exporter."
        ),
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
        "attn_sum_axis_last_min": combined_result["attn_sum_min"],
        "attn_sum_axis_last_max": combined_result["attn_sum_max"],
        "attn_sum_axis_last_mean_abs_error": combined_result["attn_abs_err_mean"],
        "attn_sum_axis_last_max_abs_error": combined_result["attn_abs_err_max"],
        "attn_sum_axis_last_within_tolerance": bool(combined_result["attn_abs_err_max"] <= args.qc_tol),
        "qc_tolerance": float(args.qc_tol),
    }

    save_array(output_dir / "parcel_modality_attn_raw.npy", combined_result["parcel_modality_attn_raw"])
    save_array(output_dir / "parcel_modality_attn_norm.npy", combined_result["parcel_modality_attn_norm"])
    save_array(output_dir / "parcel_modality_attn_per_head.npy", combined_result["parcel_modality_attn_per_head"])
    save_array(output_dir / "head_modality_attn_raw.npy", combined_result["head_modality_attn_raw"])
    save_array(output_dir / "head_modality_attn_norm.npy", combined_result["head_modality_attn_norm"])
    save_array(output_dir / "query_to_parcel.npy", combined_result["query_to_parcel"])
    save_json(output_dir / "metadata.json", combined_metadata)
    save_json(output_dir / "quality_control.json", quality_control)


def accumulate_combined_results(unit_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not unit_results:
        raise ValueError("No unit results to combine.")

    first = unit_results[0]
    num_queries = int(first["num_queries"])
    num_heads = int(first["num_heads"])
    num_modalities = int(first["raw_sum"].shape[1])

    total_samples = float(sum(result["num_samples_analyzed"] for result in unit_results))
    if total_samples <= 0:
        raise ValueError("No analyzed samples available for combined outputs.")

    total_raw_sum = np.zeros((num_queries, num_modalities), dtype=np.float64)
    total_norm_sum = np.zeros((num_queries, num_modalities), dtype=np.float64)
    total_per_head_norm_sum = np.zeros((num_heads, num_queries, num_modalities), dtype=np.float64)
    total_head_raw_sum = np.zeros((num_heads, num_modalities), dtype=np.float64)
    total_head_norm_sum = np.zeros((num_heads, num_modalities), dtype=np.float64)

    attn_sum_min = np.inf
    attn_sum_max = -np.inf
    attn_abs_err_weighted_sum = 0.0
    attn_abs_err_count = 0
    attn_abs_err_max = 0.0
    unit_summaries = []

    for result in unit_results:
        total_raw_sum += result["raw_sum"].astype(np.float64)
        total_norm_sum += result["norm_sum"].astype(np.float64)
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
        unit_summaries.append(
            {
                "unit": result["unit"],
                "num_samples_analyzed": int(result["num_samples_analyzed"]),
                "num_samples_saved": int(result["num_samples_saved"]),
                "num_heads": int(result["num_heads"]),
                "num_queries": int(result["num_queries"]),
                "num_memory_tokens": int(result["num_memory_tokens"]),
                "attn_maps_shape": result["quality_control"]["attn_maps_shape"],
            }
        )

    return {
        "num_units": len(unit_results),
        "unit_names": [result["unit"] for result in unit_results],
        "num_samples_analyzed": int(total_samples),
        "num_queries": num_queries,
        "num_heads": num_heads,
        "parcel_modality_attn_raw": (total_raw_sum / float(num_heads * total_samples)).astype(np.float32),
        "parcel_modality_attn_norm": (total_norm_sum / float(num_heads * total_samples)).astype(np.float32),
        "parcel_modality_attn_per_head": (total_per_head_norm_sum / total_samples).astype(np.float32),
        "head_modality_attn_raw": (total_head_raw_sum / float(num_queries * total_samples)).astype(np.float32),
        "head_modality_attn_norm": (total_head_norm_sum / float(num_queries * total_samples)).astype(np.float32),
        "query_to_parcel": build_query_to_parcel(num_queries),
        "attn_sum_min": attn_sum_min,
        "attn_sum_max": attn_sum_max,
        "attn_abs_err_mean": attn_abs_err_weighted_sum / max(attn_abs_err_count, 1),
        "attn_abs_err_max": attn_abs_err_max,
        "unit_summaries": unit_summaries,
    }


def main() -> None:
    args = build_parser().parse_args()
    args.modalities = normalize_modalities(args.modalities)
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
