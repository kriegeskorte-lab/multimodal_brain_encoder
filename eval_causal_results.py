from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


DEFAULT_COLUMNS = ["bourne", "figures", "life", "wolf"]
DEFAULT_SAVE_DIR = Path("./ckpt/eval_causal_results")
DEFAULT_SCAN_ROOT = Path("./ckpt")


def _resolve_no_overwrite_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    for idx in range(1, 10_000):
        candidate = path.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find an available filename for {path}")

def _resolve_result_paths(checkpoint_path: Path) -> list[Path]:
    checkpoint_dir = Path(checkpoint_path).parent
    return [
        checkpoint_dir / "test_movie_breakdown.json",
        checkpoint_dir / "test_causal_video.json",
        checkpoint_dir / "test_causal_audio.json",
        checkpoint_dir / "test_causal_text.json",
    ]


def _resolve_metrics_override_path(checkpoint_path: Path) -> Path:
    return Path(checkpoint_path).parent / "summary.json"
    # return None


def _infer_causal_type(result_path: Path) -> str:
    causal_type_by_name = {
        "test_movie_breakdown.json": "baseline",
        "test_causal_video.json": "video",
        "test_causal_audio.json": "audio",
        "test_causal_text.json": "text",
    }
    return causal_type_by_name.get(result_path.name, "")


def _safe_get(dct, *keys, default=None):
    current = dct
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _build_row(
    checkpoint_path: Path,
    summary: dict,
    columns: Iterable[str],
    *,
    causal_type: str = "",
    overall_loss_override=None,
    overall_acc_override=None,
) -> dict:
    row = {
        "checkpoint": checkpoint_path,
        "causal_type": causal_type,
        "subject": summary.get("subject"),
        "target_subject": summary.get("target_subject"),
        "test_split": summary.get("test_split"),
        "macro_loss": _safe_get(summary, "macro_average", "loss"),
        "macro_acc": _safe_get(summary, "macro_average", "acc"),
        "overall_loss": (
            overall_loss_override
            if overall_loss_override is not None
            else _safe_get(summary, "overall", "loss")
        ),
        "overall_acc": (
            overall_acc_override
            if overall_acc_override is not None
            else _safe_get(summary, "overall", "acc")
        ),
        "num_samples": _safe_get(summary, "overall", "num_samples"),
    }

    for column in columns:
        row[f"{column}_acc"] = _safe_get(summary, "per_movie", column, "acc")
        row[f"{column}_loss"] = _safe_get(summary, "per_movie", column, "loss")
        row[f"{column}_n"] = _safe_get(summary, "per_movie", column, "num_samples")

    return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate causal eval JSONs into a CSV")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help="Directory to write the output CSV into",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="eval_causal_results.csv",
        help="Output CSV filename (within --save-dir)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="If set and the output path exists, write to a new suffixed filename instead of overwriting",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    save_dir: Path = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = [
        "ckpt/1/04-07-2026-16-09/best.pt",
        "ckpt/2/04-07-2026-16-13/best.pt",
        "ckpt/3/04-08-2026-00-52/best.pt",
        "ckpt/5/04-08-2026-00-55/best.pt",
        "ckpt/1/04-18-2026-01-08/best.pt",
        "ckpt/2/04-18-2026-01-14/best.pt",
        "ckpt/3/04-18-2026-22-20/best.pt",
        "ckpt/5/04-18-2026-22-22/best.pt",
        "ckpt/1/04-08-2026-23-43/best.pt",
        "ckpt/2/04-08-2026-23-48/best.pt",
        "ckpt/3/04-08-2026-23-57/best.pt",
        "ckpt/5/04-12-2026-19-49/best.pt",
        "ckpt/1/04-19-2026-02-08/best.pt",
        "ckpt/2/04-19-2026-02-10/best.pt",
        "ckpt/3/04-20-2026-00-29/best.pt",
        "ckpt/5/04-20-2026-00-29/best.pt",
    ]

    print(f"Processing {len(checkpoint_paths)} checkpoints...")

    rows = []
    for checkpoint_path in checkpoint_paths:
        override_path = _resolve_metrics_override_path(checkpoint_path)
        if not override_path or not override_path.exists():
            continue
        with override_path.open("r") as handle:
            override_metrics = json.load(handle)

        for result_path in _resolve_result_paths(checkpoint_path):
            if not result_path.exists():
                continue

            with result_path.open("r") as handle:
                summary = json.load(handle)

            overrides = {}
            # if result_path.name == "test_movie_breakdown.json":
            #     overrides["overall_loss_override"] = override_metrics.get("test_loss")
            #     overrides["overall_acc_override"] = override_metrics.get("test_acc")

            rows.append(
                _build_row(
                    checkpoint_path,
                    summary,
                    DEFAULT_COLUMNS,
                    causal_type=_infer_causal_type(result_path),
                    **overrides,
                )
            )

    output_path = save_dir / args.output_name
    if args.no_overwrite:
        output_path = _resolve_no_overwrite_path(output_path)

    fieldnames = [
        "checkpoint",
        "run_dir",
        "causal_type",
        "subject",
        "target_subject",
        "test_split",
        "macro_loss",
        "macro_acc",
        "overall_loss",
        "overall_acc",
        "num_samples",
    ]
    for column in DEFAULT_COLUMNS:
        fieldnames.extend([f"{column}_acc", f"{column}_loss", f"{column}_n"])

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
    
