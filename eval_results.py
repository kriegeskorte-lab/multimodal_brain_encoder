from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List


DEFAULT_COLUMNS = ["bourne", "figures", "life", "wolf"]
DEFAULT_SAVE_DIR = Path("./ckpt/eval_results")
DEFAULT_SCAN_ROOT = Path("./ckpt")

def _resolve_summary_path(checkpoint_path: Path) -> Path:
    return Path(checkpoint_path).parent / "test_movie_breakdown.json"


def _safe_get(dct, *keys, default=None):
    current = dct
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _build_row(checkpoint_path: Path, summary: dict, columns: Iterable[str]) -> dict:
    row = {
        "checkpoint": checkpoint_path,
        "subject": summary.get("subject"),
        "target_subject": summary.get("target_subject"),
        "test_split": summary.get("test_split"),
        "macro_loss": _safe_get(summary, "macro_average", "loss"),
        "macro_acc": _safe_get(summary, "macro_average", "acc"),
        "overall_loss": _safe_get(summary, "overall", "loss"),
        "overall_acc": _safe_get(summary, "overall", "acc"),
        "num_samples": _safe_get(summary, "overall", "num_samples"),
    }

    for column in columns:
        row[f"{column}_acc"] = _safe_get(summary, "per_movie", column, "acc")
        row[f"{column}_loss"] = _safe_get(summary, "per_movie", column, "loss")
        row[f"{column}_n"] = _safe_get(summary, "per_movie", column, "num_samples")

    return row


def main() -> None:
    save_dir = DEFAULT_SAVE_DIR
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
        "ckpt/1/04-08-2026-13-27/best.pt",
        "ckpt/2/04-08-2026-13-29/best.pt",
        "ckpt/3/04-08-2026-13-49/best.pt",
        "ckpt/5/04-08-2026-19-18/best.pt",
        "ckpt/1/04-08-2026-19-18/best.pt",
        "ckpt/2/04-08-2026-19-19/best.pt",
        "ckpt/3/04-08-2026-21-21/best.pt",
        "ckpt/5/04-08-2026-21-30/best.pt",
        "ckpt/1/04-09-2026-11-41/best.pt",
        "ckpt/2/04-09-2026-11-43/best.pt",
        "ckpt/3/04-09-2026-11-45/best.pt",
        "ckpt/5/04-09-2026-14-20/best.pt",
        "ckpt/1/04-15-2026-19-12/best.pt",
        "ckpt/2/04-15-2026-19-10/best.pt",
        "ckpt/3/04-15-2026-19-10/best.pt",
        "ckpt/5/04-16-2026-13-45/best.pt",
        "ckpt/1/04-16-2026-13-46/best.pt",
        "ckpt/2/04-16-2026-13-46/best.pt",
        "ckpt/3/04-16-2026-23-22/best.pt",
        "ckpt/5/04-17-2026-13-49/best.pt",
        "ckpt/1/04-17-2026-11-20/best.pt",
        "ckpt/2/04-17-2026-13-53/best.pt",
        "ckpt/3/04-17-2026-15-17/best.pt",
        "ckpt/5/04-17-2026-18-52/best.pt",
        "ckpt/1/04-08-2026-23-43/best.pt",
        "ckpt/2/04-08-2026-23-48/best.pt",
        "ckpt/3/04-08-2026-23-57/best.pt",
        "ckpt/5/04-12-2026-19-49/best.pt",
        "ckpt/1/04-19-2026-02-08/best.pt",
        "ckpt/2/04-19-2026-02-10/best.pt",
        "ckpt/3/04-20-2026-00-29/best.pt",
        "ckpt/5/04-20-2026-00-29/best.pt",
        "ckpt/1/04-09-2026-14-21/best.pt",
        "ckpt/2/04-09-2026-14-52/best.pt",
        "ckpt/3/04-09-2026-16-52/best.pt",
        "ckpt/5/04-09-2026-16-56/best.pt",
        "ckpt/1/04-10-2026-10-50/best.pt",
        "ckpt/2/04-10-2026-10-51/best.pt",
        "ckpt/3/04-10-2026-17-12/best.pt",
        "ckpt/5/04-10-2026-17-13/best.pt",
        "ckpt/1/04-11-2026-12-44/best.pt",
        "ckpt/2/04-11-2026-12-46/best.pt",
        "ckpt/3/04-11-2026-12-50/best.pt",
        "ckpt/5/04-11-2026-16-23/best.pt",
        "ckpt/1/04-20-2026-04-03/best.pt",
        "ckpt/2/04-20-2026-04-03/best.pt",
        "ckpt/3/04-20-2026-12-49/best.pt",
        "ckpt/5/04-20-2026-12-52/best.pt",
        "ckpt/1/04-20-2026-23-34/best.pt",
        "ckpt/2/04-21-2026-00-50/best.pt",
        "ckpt/3/04-22-2026-11-32/best.pt",
        "ckpt/5/04-21-2026-15-23/best.pt",
        "ckpt/1/04-21-2026-22-19/best.pt",
        "ckpt/2/04-21-2026-22-20/best.pt",
        "ckpt/3/04-22-2026-01-13/best.pt",
        "ckpt/5/04-22-2026-11-33/best.pt",
        "ckpt/1/04-22-2026-16-14/best.pt", # video audio text parcels
        "ckpt/1/04-22-2026-16-17/best.pt", # video audio text parcels
        "ckpt/1/04-23-2026-00-25/best.pt", # video audio text parcels
        "ckpt/1/04-23-2026-00-28/best.pt", # video audio text parcels
        "ckpt/1/04-23-2026-13-34/best.pt", # video audio text parcels
        "ckpt/1/04-23-2026-13-35/best.pt", # video audio text parcels
        "ckpt/1/04-23-2026-23-28/best.pt", # video audio text parcels
        "ckpt/1/04-24-2026-22-15/best.pt", # video audio text parcels
        "ckpt/1/04-23-2026-13-37/best.pt", # video audio text voxels
        "ckpt/1/04-23-2026-23-33/best.pt", # video audio text voxels
        "ckpt/1/04-23-2026-23-34/best.pt", # video audio text voxels
        "ckpt/1/04-24-2026-12-08/best.pt", # video audio text voxels
        "ckpt/1/04-24-2026-12-09/best.pt", # video audio text voxels
        "ckpt/1/04-24-2026-12-10/best.pt", # video audio text voxels
        "ckpt/1/04-24-2026-20-05/best.pt", # video audio text voxels
        "ckpt/1/04-24-2026-20-50/best.pt", # video audio text voxels
    ]

    print(f"Processing {len(checkpoint_paths)} checkpoints...")

    rows = []
    for checkpoint_path in checkpoint_paths:
        summary_path = _resolve_summary_path(checkpoint_path)
        if not summary_path.exists():
            continue
        with summary_path.open("r") as handle:
            summary = json.load(handle)
        rows.append(_build_row(checkpoint_path, summary, DEFAULT_COLUMNS))

    output_path = save_dir / "eval_results.csv"
    fieldnames = [
        "checkpoint",
        "run_dir",
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
    

