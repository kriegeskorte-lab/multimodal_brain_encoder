#!/usr/bin/env python3
"""Count trainable parameters for the requested multimodal encoder settings.

This sweeps subjects 1, 2, 3, 5 across parcels vs voxels and two backbone
triplets:
  - dino / whisper / llama
  - videomae / wav2vec / deberta

The script runs a single dummy forward pass per configuration so lazy layers
are initialized before parameter counting.
"""

from __future__ import annotations

import argparse
import gc
from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from args import get_args_parser
from cneuro_dataset.cneuro_data import algonauts_dataset
from models.multimodel_backbone import BACKBONE_LIST
from models.neuro_encoder import NeuroEncoder


SUBJECTS: Sequence[int] = (1, 2, 3, 5)
READOUT_RES: Sequence[str] = ("parcels", "voxels")
BACKBONE_TRIPLETS: Sequence[tuple[str, str, str]] = (
    ("dino", "whisper", "llama"),
    ("videomae", "wav2vec", "deberta"),
)


@dataclass(frozen=True)
class CountResult:
    subject: int
    readout_res: str
    video_backbone: str
    audio_backbone: str
    text_backbone: str
    trainable_params: int
    total_params: int

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_params / self.total_params if self.total_params else 0.0


def _build_args(
    base_args: argparse.Namespace,
    subject: int,
    readout_res: str,
    triplet: tuple[str, str, str],
) -> argparse.Namespace:
    args = deepcopy(base_args)
    args.subj = subject
    args.target_subj = subject
    args.readout_res = readout_res
    args.video_backbone, args.audio_backbone, args.text_backbone = triplet
    args.modality = ["video", "audio", "text"]
    args.backbone_list = BACKBONE_LIST
    args.use_wandb = False
    args.save_checkpoints = False
    args.save_test_predictions = False
    args.pipeline_sanity_check = True
    args.lr = 0.0
    return args


def _build_test_loader(args: argparse.Namespace) -> tuple[object, DataLoader]:
    dataset = algonauts_dataset(args, include_splits=args.test_splits)
    args.valid_voxel_mask = dataset.valid_voxel_mask if args.readout_res == "voxels" else None
    args.masked_parcellation = dataset.masked_parcellation if args.readout_res == "voxels" else None

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    return dataset, loader


def _count_model_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


def count_one_configuration(
    base_args: argparse.Namespace,
    subject: int,
    readout_res: str,
    triplet: tuple[str, str, str],
) -> CountResult:
    args = _build_args(base_args, subject, readout_res, triplet)
    dataset, loader = _build_test_loader(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuroEncoder(args).to(device)
    model.eval()

    dry_samples, _ = next(iter(loader))
    with torch.inference_mode():
        _ = model(dry_samples)

    trainable_params, total_params = _count_model_parameters(model)

    del model
    del loader
    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return CountResult(
        subject=subject,
        readout_res=readout_res,
        video_backbone=triplet[0],
        audio_backbone=triplet[1],
        text_backbone=triplet[2],
        trainable_params=trainable_params,
        total_params=total_params,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count trainable parameters across subjects, readouts, and backbone triplets.")
    parser.add_argument("--subjects", nargs="*", type=int, default=list(SUBJECTS))
    parser.add_argument("--readout_res", nargs="*", choices=["parcels", "voxels"], default=list(READOUT_RES))
    parser.add_argument("--test_splits", default="movie10-ood-default", type=str)
    parser.add_argument("--ckpt_root", default="./ckpt", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--hidden_dim", default=768, type=int)
    parser.add_argument("--dim_feedforward", default=1024, type=int)
    parser.add_argument("--enc_layers", default=0, type=int)
    parser.add_argument("--dec_layers", default=1, type=int)
    parser.add_argument("--nheads", default=16, type=int)
    parser.add_argument("--num_queries", default=1000, type=int)
    parser.add_argument("--modality", nargs="+", default=["video", "audio", "text"])
    parser.add_argument("--video_backbone", default="metaclip", type=str)
    parser.add_argument("--audio_backbone", default="whisper", type=str)
    parser.add_argument("--text_backbone", default="metaclip", type=str)
    return parser.parse_args()


def main() -> None:
    base_args = get_args_parser().parse_args([])
    cli_args = parse_args()

    for attr in [
        "ckpt_root",
        "seed",
        "batch_size",
        "num_workers",
        "hidden_dim",
        "dim_feedforward",
        "enc_layers",
        "dec_layers",
        "nheads",
        "num_queries",
        "modality",
        "video_backbone",
        "audio_backbone",
        "text_backbone",
        "test_splits",
    ]:
        setattr(base_args, attr, getattr(cli_args, attr))

    print("# parameter_count")
    print(f"subjects={list(cli_args.subjects)} readout_res={list(cli_args.readout_res)}")

    results: list[CountResult] = []
    for subject in cli_args.subjects:
        for readout_res in cli_args.readout_res:
            for triplet in BACKBONE_TRIPLETS:
                print("-" * 88)
                print(
                    f"subject={subject} readout_res={readout_res} "
                    f"video={triplet[0]} audio={triplet[1]} text={triplet[2]}"
                )
                try:
                    result = count_one_configuration(base_args, subject, readout_res, triplet)
                except Exception as exc:
                    print(f"FAILED: {exc}")
                    continue

                results.append(result)
                print(f"Trainable parameters: {result.trainable_params:,} / {result.total_params:,}")
                print(f"Trainable parameters: {result.trainable_fraction:.4f}")

    if results:
        print("=" * 88)
        print("Completed parameter count sweep.")


if __name__ == "__main__":
    main()