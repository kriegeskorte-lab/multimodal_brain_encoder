from __future__ import annotations

import json
import pickle
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from args import get_args_parser
from cneuro_dataset.cneuro_data import algonauts_dataset
from models.multimodel_backbone import BACKBONE_LIST
from models.neuro_encoder import NeuroEncoder


def _apply_checkpoint_args(args, ckpt: Dict[str, object]) -> None:
    """Hydrate runtime args with training-time checkpoint args for inference compatibility."""
    ckpt_args = ckpt.get("args")
    if not isinstance(ckpt_args, dict):
        return

    preserve = {
        "resume",
        "challenge_split",
        "output_dir",
        "batch_size",
        "num_workers",
        "strict_length",
        "pipeline_sanity_check",
        "sanity_batches",
    }

    for key, value in ckpt_args.items():
        if key in preserve:
            continue
        setattr(args, key, value)


def _session_counts_from_dataset(dataset: algonauts_dataset, challenge_split: str) -> List[Tuple[str, str, int]]:
    """
    Returns list of tuples:
        (include_split_for_dataset, output_session_key, expected_num_samples)

    - For OOD: include_split is "ood_<name>", output key is "<name>".
    - For S7:  include_split is "s07e..", output key is "s07e..".
    """
    if challenge_split == "ood":
        counts = dataset._load_sample_count_file("ood")
        items = []
        for name in sorted(counts.keys()):
            items.append((f"ood_{name}", name, int(counts[name])))
        return items

    counts = dataset._load_sample_count_file("s7")
    items = []
    for split in sorted(counts.keys()):
        items.append((split, split, int(counts[split])))
    return items


def _build_loader(args, include_split: str) -> DataLoader:
    dataset = algonauts_dataset(args, include_splits=include_split)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None if args.num_workers <= 1 else 2,
    )


def _infer_session(model: NeuroEncoder, loader: DataLoader, device: torch.device, session_name: str) -> np.ndarray:
    preds: List[np.ndarray] = []
    model.eval()
    autocast_enabled = device.type == "cuda"

    with torch.inference_mode():
        for samples, _ in tqdm(
            loader,
            total=len(loader),
            desc=f"batches:{session_name}",
            leave=False,
        ):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                outputs = model(samples)
                pred = outputs["fmri_pred"]
            preds.append(pred.detach().float().cpu().numpy())

    if len(preds) == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(preds, axis=0).astype(np.float32)


def _load_model(args, ckpt: Dict[str, object], device: torch.device) -> NeuroEncoder:
    # Build one challenge dataset first so voxel metadata is available for model init.
    probe_split = "movie10-challenge-default" if args.challenge_split == "ood" else "friends-challenge-default"
    probe_dataset = algonauts_dataset(args, include_splits=probe_split)

    if args.readout_res == "voxels":
        args.valid_voxel_mask = probe_dataset.valid_voxel_mask
        args.masked_parcellation = probe_dataset.masked_parcellation
    else:
        args.valid_voxel_mask = None
        args.masked_parcellation = None

    model = NeuroEncoder(args)

    if "model" not in ckpt:
        raise KeyError(f"Checkpoint at {args.resume} does not contain key 'model'.")

    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    return model


def _save_outputs(
    output_dir: Path,
    challenge_split: str,
    subj: int,
    subject_predictions: Dict[str, np.ndarray],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = output_dir / f"{challenge_split}_pred_sub_{subj}.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(subject_predictions, f)

    subject_key = f"sub-{subj:02d}"
    submission_predictions = {subject_key: subject_predictions}

    npy_path = output_dir / f"fmri_predictions_{challenge_split}_{subject_key}.npy"
    np.save(npy_path, submission_predictions)

    zip_path = output_dir / f"fmri_predictions_{challenge_split}_{subject_key}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(npy_path, arcname=npy_path.name)

    return {
        "pkl": str(pkl_path),
        "npy": str(npy_path),
        "zip": str(zip_path),
    }


def main() -> None:
    parser = get_args_parser()
    parser.add_argument("--challenge_split", choices=["ood", "s07"], default="ood", type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--strict_length", action="store_true", help="Raise error if prediction length is shorter than expected.")
    args = parser.parse_args()

    if args.resume is None:
        raise ValueError("--resume is required, e.g. ckpt/{subj}/04-07-2026-02-27/best.pt")

    ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
    _apply_checkpoint_args(args, ckpt)
    args.backbone_list = BACKBONE_LIST

    if args.subj not in [1, 2, 3, 5]:
        raise ValueError("This script assumes subject-specific checkpoints. Use --subj in {1,2,3,5}.")

    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(args.resume).resolve().parent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a probe dataset to discover challenge sessions/counts directly from dataset logic.
    challenge_alias = "movie10-challenge-default" if args.challenge_split == "ood" else "friends-challenge-default"
    probe_dataset = algonauts_dataset(args, include_splits=challenge_alias)
    sessions = _session_counts_from_dataset(probe_dataset, args.challenge_split)
    if len(sessions) == 0:
        raise RuntimeError(f"No challenge sessions found for split={args.challenge_split}.")

    print(f"[Challenge] split={args.challenge_split} subj={args.subj} sessions={len(sessions)}")

    model = _load_model(args, ckpt, device)

    subject_predictions: Dict[str, np.ndarray] = {}
    session_summary = {}

    for include_split, output_key, expected_len in tqdm(
        sessions,
        total=len(sessions),
        desc="sessions",
    ):
        loader = _build_loader(args, include_split)
        pred = _infer_session(model, loader, device, output_key)

        got_len = int(pred.shape[0])
        if got_len > expected_len:
            pred = pred[:expected_len]
            got_len = expected_len
        elif got_len < expected_len:
            msg = (
                f"Prediction shorter than expected for {output_key}: got={got_len}, expected={expected_len}."
            )
            if args.strict_length:
                raise RuntimeError(msg)
            print(f"[WARN] {msg}")

        subject_predictions[output_key] = pred
        session_summary[output_key] = {
            "include_split": include_split,
            "expected_len": int(expected_len),
            "pred_len": int(got_len),
            "dim": int(pred.shape[1]) if pred.ndim == 2 else None,
        }
        print(f"  - {output_key:>14s} | expected={expected_len:5d} | pred={got_len:5d} | shape={tuple(pred.shape)}")

    saved = _save_outputs(
        output_dir=output_dir,
        challenge_split=args.challenge_split,
        subj=int(args.subj),
        subject_predictions=subject_predictions,
    )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "resume": str(args.resume),
        "subject": int(args.subj),
        "challenge_split": args.challenge_split,
        "num_sessions": len(sessions),
        "sessions": session_summary,
        "saved": saved,
    }
    summary_path = output_dir / f"challenge_summary_{args.challenge_split}_sub-{int(args.subj):02d}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("[Done] Saved files:")
    print(f"  pkl: {saved['pkl']}")
    print(f"  npy: {saved['npy']}")
    print(f"  zip: {saved['zip']}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
