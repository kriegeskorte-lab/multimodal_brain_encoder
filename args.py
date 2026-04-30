import argparse

def get_args_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser("multimodal encoder train/eval")

	parser.add_argument("--ckpt_root", default="./ckpt", type=str)
	parser.add_argument("--seed", default=42, type=int)
	parser.add_argument("--epochs", default=15, type=int)
	parser.add_argument("--batch_size", default=8, type=int)
	parser.add_argument("--num_workers", default=2, type=int)

	parser.add_argument("--pipeline_sanity_check", action="store_true")
	parser.add_argument("--sanity_batches", default=2, type=int)

	parser.add_argument("--lr", default=1e-4, type=float)
	parser.add_argument("--weight_decay", default=1e-4, type=float)
	parser.add_argument("--step_size", default=100, type=int)
	parser.add_argument("--step_size_gamma", default=0.1, type=float)
	parser.add_argument("--max_grad_norm", default=0.1, type=float)
	parser.add_argument("--l2_reg", default=0.0, type=float, help="Deprecated/ignored.")

	parser.add_argument("--train_splits", default="friends-train-default", type=str)
	parser.add_argument("--val_splits", default="friends-test-default", type=str)
	parser.add_argument("--test_splits", default="movie10-ood-default", type=str)

	parser.add_argument("--use_wandb", action="store_true")
	parser.add_argument("--wandb_project", default="multimodal-encoder", type=str)
	parser.add_argument("--wandb_run_name", default=None, type=str)

	parser.add_argument("--resume", default=None, type=str)
	parser.add_argument("--eval_only", action="store_true")
	parser.add_argument("--save_checkpoints", action="store_true")
	parser.add_argument("--save_test_predictions", action="store_true")
	parser.add_argument("--save_test_movie_breakdown", action="store_true")
	parser.add_argument("--save_test_causal_intervention", action="store_true")

    # Data/model hyperparameters
	parser.add_argument("--subj", "--sub", "--train_subj", dest="subj", default=1, type=int)
	parser.add_argument("--target_subj", "--eval_subj", dest="target_subj", default=1, type=int)
	parser.add_argument("--readout_res", choices=["parcels", "voxels"], default="parcels", type=str)
	parser.add_argument("--num_queries", default=1000, type=int)
	parser.add_argument("--num_frames", default=16, type=int)
	parser.add_argument("--num_parcels", default=1000, type=int)
	parser.add_argument("--num_voxels", default=122721, type=int)

	parser.add_argument("--modality", nargs="+", default=["video", "audio", "text"])
	parser.add_argument("--video_backbone", default="metaclip", type=str)
	parser.add_argument("--audio_backbone", default="whisper", type=str)
	parser.add_argument("--text_backbone", default="metaclip", type=str)

    # Transformer hyperparameters in neuro_encoder.py
	parser.add_argument("--enc_layers", default=0, type=int)
	parser.add_argument("--dec_layers", default=1, type=int)
	parser.add_argument("--dim_feedforward", default=512, type=int)
	parser.add_argument("--hidden_dim", default=256, type=int)
	parser.add_argument("--dropout", default=0.1, type=float)
	parser.add_argument("--nheads", default=8, type=int)
	parser.add_argument("--pre_norm", default=1, type=int)
	parser.add_argument("--enc_output_layer", default=-1, type=int)
	parser.add_argument("--modality_dropout", default=0.2, type=float)
	parser.add_argument("--attn_maps", action="store_true")
	parser.add_argument(
		"--attn_write_mode",
		choices=["batch", "sample"],
		default="batch",
		type=str,
		help="How to write batched attention maps to HDF5 during export.",
	)
	parser.add_argument(
		"--attn_compression",
		choices=["none", "lzf", "gzip"],
		default="lzf",
		type=str,
		help="Compression mode for exported attention maps.",
	)

	return parser
