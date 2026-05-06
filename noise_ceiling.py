from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - lets --help work in minimal system Python.
    np = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - convenience fallback for minimal environments.
    def tqdm(iterable=None, **_: Any):
        return iterable if iterable is not None else []


DEFAULT_SUBJECTS = (1, 2, 3, 5)
DEFAULT_READOUTS = ("parcels",)
DEFAULT_MOVIES = ("figures", "life")
CKPT_INFO = {
    "04-07-2026-16-09": {"subject": 1, "readout_res": "parcels", "split": "test"},
    "04-07-2026-16-13": {"subject": 2, "readout_res": "parcels", "split": "test"},
    "04-08-2026-00-52": {"subject": 3, "readout_res": "parcels", "split": "test"},
    "04-08-2026-00-55": {"subject": 5, "readout_res": "parcels", "split": "test"},
}

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/engram/nklab/datasets")


def require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "noise_ceiling.py requires h5py to read fMRI HDF5 files. "
            "Run it in the project Pixi/Jupyter environment."
        ) from exc
    return h5py


def require_numpy():
    global np
    if np is not None:
        return np
    try:
        import numpy as imported_np
    except ImportError as exc:
        raise ImportError(
            "noise_ceiling.py requires numpy. Run it in the project Pixi/Jupyter environment."
        ) from exc
    np = imported_np
    return np


@dataclass(frozen=True)
class RepeatPair:
    movie: str
    unit: str
    rep_a: str
    rep_b: str
    num_samples: int
    num_outputs: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-parcel Movie10 noise ceilings from repeated "
            "Hidden Figures and Life BOLD responses."
        )
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=list(DEFAULT_SUBJECTS))
    parser.add_argument(
        "--readouts",
        nargs="+",
        choices=list(DEFAULT_READOUTS),
        default=list(DEFAULT_READOUTS),
        help="Readout resolutions to process.",
    )
    parser.add_argument(
        "--movies",
        nargs="+",
        choices=list(DEFAULT_MOVIES),
        default=list(DEFAULT_MOVIES),
        help="Repeated Movie10 movies to use for reliability.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_REPO_ROOT / "ckpt")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REPO_ROOT / "ckpt" / "noise_ceiling_results")
    parser.add_argument(
        "--feature-chunk-size",
        type=int,
        default=8192,
        help="Number of parcels to process per chunk.",
    )
    parser.add_argument(
        "--normalize-existing-acc",
        action="store_true",
        help=(
            "Normalize only the selected CKPT_INFO test_<readout>_acc.npy files and save "
            "test_<readout>_normalized_acc.npy in each checkpoint directory."
        ),
    )
    parser.add_argument(
        "--acc-files",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Optional explicit model accuracy vectors to normalize. Filenames must contain "
            "test_parcels_acc.npy and be under ckpt/<subject>/<run>/ "
            "or use --acc-subject/--acc-readout for a single file."
        ),
    )
    parser.add_argument("--acc-subject", type=int, default=None)
    parser.add_argument("--acc-readout", choices=list(DEFAULT_READOUTS), default=None)
    parser.add_argument(
        "--rho-max-floor",
        type=float,
        default=1e-6,
        help="Minimum positive ceiling required before computing rho_norm.",
    )
    return parser


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


def save_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_ready(row))


def parcel_movie10_path(data_root: Path, subject: int) -> Path:
    return (
        data_root
        / "algonauts_2025.competitors"
        / "fmri"
        / f"sub-0{subject}"
        / "func"
        / f"sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5"
    )


def normalize_dataset_name(name: str) -> str:
    name = str(name)
    if name.startswith("task-movie10_"):
        return name[len("task-movie10_") :]
    if name.startswith("task-friends_"):
        return name[len("task-friends_") :]
    return name


def repeat_unit(name: str, movies: Sequence[str]) -> tuple[str, str] | None:
    normalized = normalize_dataset_name(name).lower()
    for movie in movies:
        match = re.search(rf"({re.escape(movie.lower())}\d{{2}})", normalized)
        if match is not None:
            return movie.lower(), match.group(1)
    return None


def sorted_repeat_names(names: Iterable[str], movies: Sequence[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for name in names:
        parsed = repeat_unit(name, movies)
        if parsed is None:
            continue
        _, unit = parsed
        grouped.setdefault(unit, []).append(str(name))
    return {unit: sorted(unit_names) for unit, unit_names in sorted(grouped.items())}


def collect_parcel_pairs(h5f: h5py.File, movies: Sequence[str]) -> List[RepeatPair]:
    grouped = sorted_repeat_names(h5f.keys(), movies)
    pairs: List[RepeatPair] = []
    for unit, names in grouped.items():
        if len(names) < 2:
            continue
        rep_a, rep_b = names[:2]
        ds_a = h5f[rep_a]
        ds_b = h5f[rep_b]
        if ds_a.ndim != 2 or ds_b.ndim != 2:
            raise ValueError(f"Expected 2D parcel datasets for {unit}, got {ds_a.shape} and {ds_b.shape}")
        if int(ds_a.shape[1]) != int(ds_b.shape[1]):
            raise ValueError(f"Output dimension mismatch for {unit}: {ds_a.shape} vs {ds_b.shape}")
        movie = repeat_unit(unit, movies)[0]  # type: ignore[index]
        pairs.append(
            RepeatPair(
                movie=movie,
                unit=unit,
                rep_a=rep_a,
                rep_b=rep_b,
                num_samples=min(int(ds_a.shape[0]), int(ds_b.shape[0])),
                num_outputs=int(ds_a.shape[1]),
            )
        )
    return pairs


def chunk_bounds(size: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, size, chunk_size):
        yield start, min(start + chunk_size, size)


def pearson_columns(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    count = finite.sum(axis=0, dtype=np.float64)
    x = np.where(finite, x, 0.0)
    y = np.where(finite, y, 0.0)

    sum_x = x.sum(axis=0)
    sum_y = y.sum(axis=0)
    sum_x2 = (x * x).sum(axis=0)
    sum_y2 = (y * y).sum(axis=0)
    sum_xy = (x * y).sum(axis=0)

    corr = np.full(x.shape[1], np.nan, dtype=np.float64)
    valid = count > 1
    cov = np.full(x.shape[1], np.nan, dtype=np.float64)
    var_x = np.full(x.shape[1], np.nan, dtype=np.float64)
    var_y = np.full(x.shape[1], np.nan, dtype=np.float64)
    cov[valid] = sum_xy[valid] - (sum_x[valid] * sum_y[valid]) / count[valid]
    var_x[valid] = sum_x2[valid] - (sum_x[valid] * sum_x[valid]) / count[valid]
    var_y[valid] = sum_y2[valid] - (sum_y[valid] * sum_y[valid]) / count[valid]
    denom = np.sqrt(np.maximum(var_x, 0.0) * np.maximum(var_y, 0.0))
    ok = valid & np.isfinite(denom) & (denom > 0.0)
    corr[ok] = cov[ok] / denom[ok]
    return np.clip(corr, -1.0, 1.0).astype(np.float32), count.astype(np.int32)


def accumulate_pair_stats(
    pairs: Sequence[RepeatPair],
    get_dataset,
    num_outputs: int,
    feature_chunk_size: int,
) -> Dict[str, np.ndarray]:
    rho_self_by_pair = np.full((len(pairs), num_outputs), np.nan, dtype=np.float32)
    n_obs_by_pair = np.zeros((len(pairs), num_outputs), dtype=np.int32)

    for start, end in tqdm(
        list(chunk_bounds(num_outputs, feature_chunk_size)),
        desc="Feature chunks",
        unit="chunk",
        leave=False,
    ):
        for pair_idx, pair in enumerate(pairs):
            if pair.num_samples <= 1:
                continue
            ds_a = get_dataset(pair.rep_a)
            ds_b = get_dataset(pair.rep_b)
            x = np.asarray(ds_a[: pair.num_samples, start:end], dtype=np.float64)
            y = np.asarray(ds_b[: pair.num_samples, start:end], dtype=np.float64)
            corr, count = pearson_columns(x, y)
            rho_self_by_pair[pair_idx, start:end] = corr
            n_obs_by_pair[pair_idx, start:end] = count

    rho_sb_by_pair = spearman_brown(rho_self_by_pair)
    rho_sb_by_pair[~np.isfinite(rho_sb_by_pair) | (rho_sb_by_pair <= 0.0)] = np.nan
    rho_max_by_pair = np.sqrt(rho_sb_by_pair).astype(np.float32)

    rho_self = np.nanmean(rho_self_by_pair, axis=0).astype(np.float32)
    rho_sb = np.nanmean(rho_sb_by_pair, axis=0).astype(np.float32)
    rho_max = np.nanmean(rho_max_by_pair, axis=0).astype(np.float32)
    n_obs = np.sum(n_obs_by_pair, axis=0, dtype=np.int32)
    return {
        "rho_self": rho_self,
        "rho_sb": rho_sb,
        "rho_max": rho_max,
        "n_obs": n_obs,
        "rho_self_by_pair": rho_self_by_pair,
        "rho_sb_by_pair": rho_sb_by_pair,
        "rho_max_by_pair": rho_max_by_pair,
        "n_obs_by_pair": n_obs_by_pair,
    }


def spearman_brown(rho_self: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho_self, dtype=np.float32)
    out = np.full_like(rho, np.nan, dtype=np.float32)
    valid = np.isfinite(rho) & (rho > -1.0)
    out[valid] = (2.0 * rho[valid]) / (1.0 + rho[valid])
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def pair_rows(pairs: Sequence[RepeatPair]) -> List[Dict[str, Any]]:
    return [
        {
            "movie": pair.movie,
            "unit": pair.unit,
            "rep_a": pair.rep_a,
            "rep_b": pair.rep_b,
            "num_samples": pair.num_samples,
            "num_outputs": pair.num_outputs,
        }
        for pair in pairs
    ]


def add_movie_level_stats(stats: Dict[str, np.ndarray], pairs: Sequence[RepeatPair]) -> None:
    movies = sorted({pair.movie for pair in pairs})
    movie_rho_self = np.full((len(movies), stats["rho_self"].shape[0]), np.nan, dtype=np.float32)
    movie_rho_sb = np.full_like(movie_rho_self, np.nan)
    movie_rho_max = np.full_like(movie_rho_self, np.nan)
    movie_n_obs = np.zeros((len(movies), stats["rho_self"].shape[0]), dtype=np.int32)

    for movie_idx, movie in enumerate(movies):
        pair_indices = [idx for idx, pair in enumerate(pairs) if pair.movie == movie]
        movie_rho_self[movie_idx] = np.nanmean(stats["rho_self_by_pair"][pair_indices], axis=0)
        movie_rho_sb[movie_idx] = np.nanmean(stats["rho_sb_by_pair"][pair_indices], axis=0)
        movie_rho_max[movie_idx] = np.nanmean(stats["rho_max_by_pair"][pair_indices], axis=0)
        movie_n_obs[movie_idx] = np.sum(stats["n_obs_by_pair"][pair_indices], axis=0, dtype=np.int32)

    stats["movie_names"] = np.asarray(movies, dtype=object)
    stats["rho_self_by_movie"] = movie_rho_self
    stats["rho_sb_by_movie"] = movie_rho_sb
    stats["rho_max_by_movie"] = movie_rho_max
    stats["n_obs_by_movie"] = movie_n_obs


def summarize_vector(name: str, values: np.ndarray) -> Dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"{name}_finite": 0}
    return {
        f"{name}_finite": int(finite.size),
        f"{name}_mean": float(np.mean(finite)),
        f"{name}_median": float(np.median(finite)),
        f"{name}_std": float(np.std(finite)),
        f"{name}_min": float(np.min(finite)),
        f"{name}_max": float(np.max(finite)),
    }


def save_ceiling_outputs(
    output_dir: Path,
    subject: int,
    readout: str,
    source_path: Path,
    pairs: Sequence[RepeatPair],
    stats: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, value in stats.items():
        if key == "movie_names":
            np.save(output_dir / f"{key}.npy", value)
        elif key.startswith("n_obs"):
            np.save(output_dir / f"{key}.npy", value.astype(np.int32, copy=False))
        else:
            np.save(output_dir / f"{key}.npy", value.astype(np.float32, copy=False))

    rows = pair_rows(pairs)
    save_csv(
        output_dir / "repeat_pairs.csv",
        rows,
        fieldnames=["movie", "unit", "rep_a", "rep_b", "num_samples", "num_outputs"],
    )
    metadata = {
        "subject": subject,
        "readout": readout,
        "source_path": source_path,
        "movies": list(args.movies),
        "num_pairs": len(pairs),
        "num_outputs": int(stats["rho_self"].shape[0]),
        "formula": {
            "rho_self_by_pair": "Pearson correlation between repeat A and repeat B across matched TRs for each repeated movie unit.",
            "rho_self": "Mean rho_self_by_pair across repeated movie units.",
            "rho_sb_by_pair": "2 * rho_self_by_pair / (1 + rho_self_by_pair). Nonpositive values are set to NaN before ceilings.",
            "rho_sb": "Mean rho_sb_by_pair across repeated movie units.",
            "rho_max": "Mean sqrt(rho_sb_by_pair) across repeated movie units after setting nonpositive rho_sb_by_pair to NaN.",
            "rho_norm": "model rho / rho_max where rho_max > rho_max_floor.",
        },
        "feature_chunk_size": int(args.feature_chunk_size),
        "rho_max_floor": float(args.rho_max_floor),
        "summaries": {
            **summarize_vector("rho_self", stats["rho_self"]),
            **summarize_vector("rho_sb", stats["rho_sb"]),
            **summarize_vector("rho_max", stats["rho_max"]),
        },
    }
    save_json(output_dir / "metadata.json", metadata)


def compute_subject_readout(subject: int, readout: str, args: argparse.Namespace) -> Dict[str, Any]:
    require_numpy()
    h5py = require_h5py()
    out_dir = args.output_root / f"sub-{subject:02d}" / readout
    if readout != "parcels":
        raise ValueError(f"Unsupported readout: {readout}")
    source_path = parcel_movie10_path(args.data_root, subject)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    with h5py.File(source_path, "r") as h5f:
        pairs = collect_parcel_pairs(h5f, args.movies)
        if not pairs:
            raise ValueError(f"No repeat pairs found in {source_path}")
        num_outputs = pairs[0].num_outputs
        stats = accumulate_pair_stats(
            pairs=pairs,
            get_dataset=lambda name: h5f[name],
            num_outputs=num_outputs,
            feature_chunk_size=args.feature_chunk_size,
        )

    add_movie_level_stats(stats, pairs)
    save_ceiling_outputs(out_dir, subject, readout, source_path, pairs, stats, args)
    return {
        "subject": subject,
        "readout": readout,
        "output_dir": out_dir,
        "source_path": source_path,
        "num_pairs": len(pairs),
        "num_outputs": int(stats["rho_self"].shape[0]),
        **summarize_vector("rho_max", stats["rho_max"]),
    }


def infer_acc_subject_readout(path: Path, args: argparse.Namespace) -> tuple[int, str]:
    if args.acc_subject is not None and args.acc_readout is not None:
        return int(args.acc_subject), str(args.acc_readout)

    readout = None
    if path.name.endswith("_parcels_acc.npy"):
        readout = "parcels"
    if readout is None:
        raise ValueError(f"Cannot infer parcel readout from accuracy filename: {path}")

    parts = path.resolve().parts
    for idx, part in enumerate(parts):
        if part == "ckpt" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1]), readout
            except ValueError:
                break
    raise ValueError(
        f"Cannot infer subject from {path}. Place file under ckpt/<subject>/<run>/ "
        "or pass --acc-subject and --acc-readout."
    )


def discover_acc_files(args: argparse.Namespace) -> List[Path]:
    explicit = [Path(p) for p in args.acc_files] if args.acc_files else []
    if explicit:
        return explicit
    if not args.normalize_existing_acc:
        return []
    files: List[Path] = []
    requested_subjects = set(int(subject) for subject in args.subjects)
    requested_readouts = set(str(readout) for readout in args.readouts)
    for run_name, info in CKPT_INFO.items():
        subject = int(info["subject"])
        readout = str(info["readout_res"])
        split = str(info.get("split", "test"))
        if subject not in requested_subjects or readout not in requested_readouts:
            continue
        files.append(args.ckpt_root / str(subject) / run_name / f"{split}_{readout}_acc.npy")
    return files


def normalize_accuracy_file(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    require_numpy()
    subject, readout = infer_acc_subject_readout(path, args)
    ceiling_path = args.output_root / f"sub-{subject:02d}" / readout / "rho_max.npy"
    if not ceiling_path.exists():
        raise FileNotFoundError(f"Missing ceiling vector for {path}: {ceiling_path}")

    acc = np.load(path).astype(np.float32, copy=False)
    rho_max = np.load(ceiling_path).astype(np.float32, copy=False)
    if acc.shape != rho_max.shape:
        raise ValueError(f"Shape mismatch for {path}: acc={acc.shape}, rho_max={rho_max.shape}")

    rho_norm = np.full_like(acc, np.nan, dtype=np.float32)
    valid = np.isfinite(acc) & np.isfinite(rho_max) & (rho_max > args.rho_max_floor)
    rho_norm[valid] = acc[valid] / rho_max[valid]
    rho_norm_clipped = np.clip(rho_norm, -1.0, 1.0).astype(np.float32)

    rel_name = "_".join(path.with_suffix("").parts[-3:])
    metadata_dir = args.output_root / f"sub-{subject:02d}" / readout / "normalized_model_acc"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    norm_path = path.parent / f"test_{readout}_normalized_acc.npy"
    clipped_path = metadata_dir / f"{rel_name}_rho_norm_clipped.npy"
    np.save(norm_path, rho_norm)
    np.save(clipped_path, rho_norm_clipped)

    metadata = {
        "source_acc": path,
        "ceiling": ceiling_path,
        "subject": subject,
        "readout": readout,
        "rho_max_floor": args.rho_max_floor,
        "rho_norm_path": norm_path,
        "rho_norm_clipped_path": clipped_path,
        "summaries": {
            **summarize_vector("acc", acc),
            **summarize_vector("rho_norm", rho_norm),
            **summarize_vector("rho_norm_clipped", rho_norm_clipped),
        },
    }
    save_json(metadata_dir / f"{rel_name}_metadata.json", metadata)
    return {
        "source_acc": path,
        "subject": subject,
        "readout": readout,
        "rho_norm_path": norm_path,
        **metadata["summaries"],
    }


def main() -> None:
    args = build_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    for subject in tqdm(args.subjects, desc="Subjects", unit="subject"):
        for readout in args.readouts:
            print(f"Computing noise ceiling: sub-{subject:02d} {readout}")
            row = compute_subject_readout(subject, readout, args)
            summary_rows.append(row)
            print(
                f"  saved={row['output_dir']} outputs={row['num_outputs']} "
                f"pairs={row['num_pairs']} rho_max_mean={row.get('rho_max_mean', float('nan')):.4f}"
            )

    save_csv(
        args.output_root / "noise_ceiling_summary.csv",
        summary_rows,
        fieldnames=[
            "subject",
            "readout",
            "output_dir",
            "source_path",
            "num_pairs",
            "num_outputs",
            "rho_max_finite",
            "rho_max_mean",
            "rho_max_median",
            "rho_max_std",
            "rho_max_min",
            "rho_max_max",
        ],
    )
    save_json(
        args.output_root / "manifest.json",
        {
            "subjects": args.subjects,
            "readouts": args.readouts,
            "movies": args.movies,
            "output_root": args.output_root,
            "summary_csv": args.output_root / "noise_ceiling_summary.csv",
        },
    )

    acc_files = discover_acc_files(args)
    if acc_files:
        norm_rows = []
        for path in tqdm(acc_files, desc="Normalize model acc", unit="file"):
            norm_rows.append(normalize_accuracy_file(path, args))
        save_csv(
            args.output_root / "normalized_model_acc_summary.csv",
            norm_rows,
            fieldnames=[
                "source_acc",
                "subject",
                "readout",
                "rho_norm_path",
                "acc_finite",
                "acc_mean",
                "acc_median",
                "acc_std",
                "acc_min",
                "acc_max",
                "rho_norm_finite",
                "rho_norm_mean",
                "rho_norm_median",
                "rho_norm_std",
                "rho_norm_min",
                "rho_norm_max",
                "rho_norm_clipped_finite",
                "rho_norm_clipped_mean",
                "rho_norm_clipped_median",
                "rho_norm_clipped_std",
                "rho_norm_clipped_min",
                "rho_norm_clipped_max",
            ],
        )


if __name__ == "__main__":
    main()
