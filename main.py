from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import args
import torch
import numpy as np

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs
ddp_kwargs = DistributedDataParallelKwargs(
    broadcast_buffers=False,   # avoids pre-forward _sync_buffers hangs
)

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from eval import evaluate
from metric import mse_loss
from args import get_args_parser
from train import train_one_epoch
from models.neuro_encoder import NeuroEncoder
# from cneuro_dataset.cneuro_data_ethan import algonauts_dataset
from models.multimodel_backbone import BACKBONE_LIST
from cneuro_dataset.cneuro_data import algonauts_dataset

import warnings
warnings.filterwarnings("ignore")

class MSECriterion(nn.Module):
	"""Pure MSE criterion. Ignores outputs['l2_reg'] by design."""

	def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
		if targets is None:
			raise ValueError("Targets are required for MSE training/evaluation.")
		return mse_loss(outputs["fmri_pred"], targets)


def _subset_for_sanity(dataset, max_batches: int, batch_size: int):
	max_items = min(len(dataset), max_batches * batch_size)
	return Subset(dataset, list(range(max_items)))


def build_dataloaders(args) -> Dict[str, DataLoader]:
	train_dataset = algonauts_dataset(args, include_splits=args.train_splits)
	val_dataset = algonauts_dataset(args, include_splits=args.val_splits)
	test_dataset = algonauts_dataset(args, include_splits=args.test_splits)
	
	# set parcellation/masked_parcellation attributes on args for use in model initialization
	args.valid_voxel_mask = test_dataset.valid_voxel_mask if args.readout_res == "voxels" else None
	args.masked_parcellation = test_dataset.masked_parcellation if args.readout_res == "voxels" else None
	

	if args.pipeline_sanity_check:
		train_dataset = _subset_for_sanity(train_dataset, args.sanity_batches, args.batch_size)
		val_dataset = _subset_for_sanity(val_dataset, args.sanity_batches, args.batch_size)
		test_dataset = _subset_for_sanity(test_dataset, args.sanity_batches, args.batch_size)

	common = {
		"batch_size": args.batch_size,
		"num_workers": args.num_workers,
		"pin_memory": True,
		# "persistent_workers": args.num_workers > 0,
		"persistent_workers": False,
		"prefetch_factor": None if args.num_workers == 0 else 1,  # default is 2, 
	}

	train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **common)
	val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
	test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **common)

	return {
		"train": train_loader,
		"val": val_loader,
		"test": test_loader,
	}

def main() -> None:
	parser = get_args_parser()
	args = parser.parse_args()

	if len(args.modality) == 0:
		raise ValueError("No valid modalities selected. Use any of: video audio text")

	args.backbone_list = BACKBONE_LIST
	args.save_checkpoints = True
	args.save_test_predictions = True

	if args.pipeline_sanity_check:
		args.lr = 0.0
		args.use_wandb = False
		args.save_checkpoints = False
		args.save_test_predictions = False

	output_dir = None
	if args.resume is not None:
		output_dir = Path(args.resume).parent
	elif not args.pipeline_sanity_check:
		time_tag = datetime.now().strftime("%m-%d-%Y-%H-%M")
		output_dir = Path(args.ckpt_root) / str(args.subj) / time_tag
		output_dir.mkdir(parents=True, exist_ok=True)

	mp_mode = "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "fp16"
	# mp_mode = "fp16"
	accelerator = Accelerator(
		log_with="wandb" if args.use_wandb else None, 
		kwargs_handlers=[ddp_kwargs],
		mixed_precision=mp_mode,
	)
	# accelerator = Accelerator(log_with="wandb" if args.use_wandb else None)
	set_seed(args.seed)
	accelerator.print(f"Using mixed precision: {accelerator.mixed_precision}")

	accelerator.print("Arguments:")
	for k, v in vars(args).items():
		if k in ["subj", "batch_size", "lr", "readout_res", "step_size", "step_size_gamma", "modality_dropout", "modality", "video_backbone", "audio_backbone", "text_backbone"]:
			accelerator.print(f"  {k}: {v}")

	if args.use_wandb:
		accelerator.init_trackers(
			project_name=args.wandb_project,
			config=vars(args),
			init_kwargs={"wandb": {"name": args.wandb_run_name + f'_{time_tag}' if args.wandb_run_name else time_tag}},
		)

	dataloaders = build_dataloaders(args)
	accelerator.print(f"Data loaders built. Train batches: {len(dataloaders['train'])}, Val batches: {len(dataloaders['val'])}, Test batches: {len(dataloaders['test'])}")

	model = NeuroEncoder(args)
    # run a dummy forward pass to initialize weights before wrapping with accelerator for proper device placement
	dry_samples, _ = next(iter(dataloaders["test"]))
	# print(f"Dry sample video shape: {dry_samples['video']['pixel_values'].shape}") # torch.Size([2, 16, 3, 224, 224])
	with torch.no_grad(), accelerator.autocast():
		model = model.to(accelerator.device)
		model.eval()
		_ = model(dry_samples)
		model.train()
	
	criterion = MSECriterion()
	optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	# scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.step_size_gamma)
	scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)  # alternative scheduler

	start_epoch = 0
	best_val_acc = float("-inf")

	if args.resume:
		ckpt = torch.load(args.resume, map_location="cpu")
		model.load_state_dict(ckpt["model"], strict=False)
		if not args.eval_only and "optimizer" in ckpt:
			optimizer.load_state_dict(ckpt["optimizer"])
		if not args.eval_only and "scheduler" in ckpt and ckpt["scheduler"] is not None:
			scheduler.load_state_dict(ckpt["scheduler"])
		start_epoch = int(ckpt.get("epoch", -1)) + 1
		best_val_acc = float(ckpt.get("best_val_acc", best_val_acc))

	model, optimizer, dataloaders["train"], dataloaders["val"], dataloaders["test"] = accelerator.prepare(
		model,
		optimizer,
		dataloaders["train"],
		dataloaders["val"],
		dataloaders["test"]
	)

	max_batches = args.sanity_batches if args.pipeline_sanity_check else None
	best_ckpt_path = output_dir / "best.pt" if output_dir else None

	if not args.eval_only:
		for epoch in range(start_epoch, args.epochs):
			train_stats = train_one_epoch(
				model=model,
				criterion=criterion,
				data_loader=dataloaders["train"],
				optimizer=optimizer,
				accelerator=accelerator,
				epoch=epoch,
				target_subj=args.subj,
				max_grad_norm=args.max_grad_norm,
				max_batches=max_batches,
			)

			val_stats = evaluate(
				model=model,
				criterion=criterion,
				data_loader=dataloaders["val"],
				accelerator=accelerator,
				target_subj=args.target_subj,
				split_name="val",
				return_predictions=False,
				max_batches=max_batches,
			)

			scheduler.step()

			epoch_stats = {
				"epoch": epoch,
				**train_stats,
				**val_stats,
				"best_val_acc": max(best_val_acc, float(val_stats["val_acc"])),
			}

			accelerator.print(
				f"Epoch {epoch:03d} \n"
				f"Train : loss={epoch_stats['train_loss']:.4f} | acc={epoch_stats['train_acc']:.4f} \n"
				f"Val   : loss={epoch_stats['val_loss']:.4f} | acc={epoch_stats['val_acc']:.4f}"
			)

			if args.use_wandb:
				accelerator.log(epoch_stats, step=epoch)

			improved = float(val_stats["val_acc"]) > best_val_acc
			if improved:
				best_val_acc = float(val_stats["val_acc"])

				if args.save_checkpoints and accelerator.is_main_process:
					ckpt = {
						"epoch": epoch,
						"model": accelerator.unwrap_model(model).state_dict(),
						"optimizer": optimizer.state_dict(),
						"scheduler": scheduler.state_dict(),
						"best_val_acc": best_val_acc,
						"args": vars(args),
					}
					torch.save(ckpt, best_ckpt_path)

		accelerator.wait_for_everyone()

	if args.save_checkpoints and best_ckpt_path is not None and best_ckpt_path.exists():
		accelerator.print(f"Loading best checkpoint from {best_ckpt_path}")
		best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
		accelerator.unwrap_model(model).load_state_dict(best_ckpt["model"], strict=False)
		accelerator.print(f"Best val acc from checkpoint {best_ckpt['epoch']}: {best_ckpt['best_val_acc']}")

	test_stats = evaluate(
		model=model,
		criterion=criterion,
		data_loader=dataloaders["test"],
		accelerator=accelerator,
		target_subj=args.target_subj,
		split_name="test",
		return_predictions=args.save_test_predictions,
		max_batches=max_batches,
	)
	test_loss = test_stats.get("test_loss")
	test_acc = test_stats.get("test_acc")
	test_loss_str = f"{float(test_loss):.6f}" if test_loss is not None else "None"
	test_acc_str = f"{float(test_acc):.6f}" if test_acc is not None else "None"

	accelerator.print(
		f"==================================\nTest  : loss={test_loss_str} | acc={test_acc_str}"
	)

	if (
		args.save_test_predictions
		and output_dir is not None
		and accelerator.is_main_process
		and test_stats.get("preds") is not None
	):
		pred_path = output_dir / "pred_test.npy"
		with pred_path.open("wb") as f:
			np.save(f, test_stats["preds"])

	if accelerator.is_main_process and output_dir is not None:
		summary = {
			"best_val_acc": best_val_acc,
			"test_loss": float(test_loss) if test_loss is not None else None,
			"test_acc": float(test_acc) if test_acc is not None else None,
		}
		with (output_dir / "summary.json").open("w") as f:
			json.dump(summary, f, indent=2)

	if args.use_wandb:
		accelerator.wait_for_everyone()
		accelerator.end_training()


if __name__ == "__main__":
	main()
