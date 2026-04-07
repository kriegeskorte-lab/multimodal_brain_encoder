from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch import nn
from torch.utils.data import DataLoader, Subset

from args import get_args_parser
from cneuro_dataset.cneuro_data import SPLIT_GROUP_ALIASES, algonauts_dataset
from eval import evaluate
from metric import mse_loss
from models.multimodel_backbone import BACKBONE_LIST
from models.neuro_encoder import NeuroEncoder

import warnings
warnings.filterwarnings("ignore")


ddp_kwargs = DistributedDataParallelKwargs(
    broadcast_buffers=False,
)


class MSECriterion(nn.Module):
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        if targets is None:
            raise ValueError("Targets are required for evaluation.")
        return mse_loss(outputs["fmri_pred"], targets)


def _subset_for_sanity(dataset, max_batches: int, batch_size: int):
    max_items = min(len(dataset), max_batches * batch_size)
    return Subset(dataset, list(range(max_items)))


def _build_test_loader(args, split_spec: str) -> DataLoader:
    dataset = algonauts_dataset(args, include_splits=split_spec)

    if args.pipeline_sanity_check:
        dataset = _subset_for_sanity(dataset, args.sanity_batches, args.batch_size)

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


def _safe_float(x):
    return float(x) if x is not None else None


def _fmt(x):
    return f"{x:.6f}" if x is not None else "None"


def _normalize_split_spec(spec):
    if spec is None:
        return []
    if isinstance(spec, str):
        return [tok.strip().lower() for tok in spec.split(",") if tok.strip()]
    if isinstance(spec, (list, tuple, set)):
        return [str(tok).strip().lower() for tok in spec if str(tok).strip()]
    raise TypeError(f"Unsupported split specification type: {type(spec)}")


def _expand_split_tokens(tokens: List[str]) -> List[str]:
    expanded: List[str] = []

    def _expand(tok: str):
        if tok in SPLIT_GROUP_ALIASES:
            for child in SPLIT_GROUP_ALIASES[tok]:
                _expand(str(child).lower())
        else:
            expanded.append(tok)

    for token in tokens:
        _expand(token)

    # preserve order and remove duplicates
    return list(dict.fromkeys(expanded))


def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()

    args.backbone_list = BACKBONE_LIST
    set_seed(args.seed)

    if args.resume is None:
        raise ValueError("--resume is required for test.py")

    output_dir = Path(args.resume).parent

    mp_mode = "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "fp16"
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=mp_mode)

    criterion = MSECriterion()
    model = NeuroEncoder(args)

    ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    accelerator.print(f"Best checkpoint loaded from {ckpt['epoch']} with best_val_acc={ckpt.get('best_val_acc', 'N/A')}")

    test_tokens = _normalize_split_spec(args.test_splits)
    eval_units = _expand_split_tokens(test_tokens)
    if len(eval_units) == 0:
        raise ValueError(f"No evaluation units resolved from test_splits={args.test_splits}")

    unit_loaders = {unit: _build_test_loader(args, unit) for unit in eval_units}
    all_test_loader = _build_test_loader(args, args.test_splits)

    prepared = accelerator.prepare(model, *unit_loaders.values(), all_test_loader)
    model = prepared[0]

    prepared_unit_loaders = dict(zip(eval_units, prepared[1 : 1 + len(eval_units)]))
    prepared_all_loader = prepared[-1]

    accelerator.print(f"Using mixed precision: {accelerator.mixed_precision}")
    accelerator.print(f"Evaluating checkpoint: {args.resume}")

    per_movie: Dict[str, Dict[str, float]] = {}
    movie_losses: List[float] = []
    movie_accs: List[float] = []

    for movie in eval_units:
        stats = evaluate(
            model=model,
            criterion=criterion,
            data_loader=prepared_unit_loaders[movie],
            accelerator=accelerator,
            target_subj=args.target_subj,
            split_name="test",
            return_predictions=False,
            max_batches=args.sanity_batches if args.pipeline_sanity_check else None,
        )

        movie_loss = _safe_float(stats.get("test_loss"))
        movie_acc = _safe_float(stats.get("test_acc"))

        if movie_loss is not None:
            movie_losses.append(movie_loss)
        if movie_acc is not None:
            movie_accs.append(movie_acc)

        per_movie[movie] = {
            "loss": movie_loss,
            "acc": movie_acc,
            "num_samples": len(unit_loaders[movie].dataset),
        }

        accelerator.print(
            f"Unit={movie:>7s} | loss={_fmt(movie_loss)} | acc={_fmt(movie_acc)} | n={len(unit_loaders[movie].dataset)}"
        )

    macro_avg = {
        "loss": float(np.mean(movie_losses)) if movie_losses else None,
        "acc": float(np.mean(movie_accs)) if movie_accs else None,
    }

    overall_stats = evaluate(
        model=model,
        criterion=criterion,
        data_loader=prepared_all_loader,
        accelerator=accelerator,
        target_subj=args.target_subj,
        split_name="test",
        return_predictions=False,
        max_batches=args.sanity_batches if args.pipeline_sanity_check else None,
    )

    overall = {
        "loss": _safe_float(overall_stats.get("test_loss")),
        "acc": _safe_float(overall_stats.get("test_acc")),
        "split": args.test_splits,
        "num_samples": len(all_test_loader.dataset),
    }

    accelerator.print("=" * 60)
    accelerator.print(
        f"Macro average across movies | loss={_fmt(macro_avg['loss'])} | acc={_fmt(macro_avg['acc'])}"
    )
    accelerator.print(
        f"Overall ({args.test_splits})     | loss={_fmt(overall['loss'])} | acc={_fmt(overall['acc'])}"
    )

    if accelerator.is_main_process:
        summary = {
            "checkpoint": args.resume,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "subject": int(args.subj),
            "target_subject": int(args.target_subj),
            "test_split": args.test_splits,
            "eval_units": eval_units,
            "per_movie": per_movie,
            "macro_average": macro_avg,
            "overall": overall,
        }
        out_path = output_dir / "test_movie_breakdown.json"
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2)
        accelerator.print(f"Saved per-movie summary to {out_path}")


if __name__ == "__main__":
    main()
