from __future__ import annotations

import json
import queue
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import sys
sys.path.append("/engram/nklab/pf2477")

from multimodal_encoder.args import get_args_parser
from multimodal_encoder.cneuro_dataset.cneuro_data import SPLIT_GROUP_ALIASES, algonauts_dataset
from multimodal_encoder.models.multimodel_backbone import BACKBONE_LIST
from multimodal_encoder.models.neuro_encoder import NeuroEncoder

warnings.filterwarnings("ignore")


ddp_kwargs = DistributedDataParallelKwargs(
    broadcast_buffers=False,
)


def _subset_for_sanity(dataset, max_batches: int, batch_size: int):
    max_items = min(len(dataset), max_batches * batch_size)
    return Subset(dataset, list(range(max_items)))


def _build_test_loader(args, split_spec: str) -> DataLoader:
    dataset = algonauts_dataset(args, include_splits=split_spec)
    args.valid_voxel_mask = dataset.valid_voxel_mask if args.readout_res == "voxels" else None
    args.masked_parcellation = dataset.masked_parcellation if args.readout_res == "voxels" else None

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


def _to_string_list(value: Any, batch_size: int) -> List[str]:
    if isinstance(value, str):
        return [value] * batch_size
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist()]
    return [str(value)] * batch_size


def _to_int_list(value: Any, batch_size: int) -> List[int]:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return [int(value.item())] * batch_size
        return [int(x) for x in value.detach().cpu().tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    if isinstance(value, np.ndarray):
        return [int(x) for x in value.tolist()]
    return [int(value)] * batch_size


def _extract_batch_metadata(samples: Dict[str, Any], batch_size: int) -> tuple[List[str], List[int]]:
    splits = _to_string_list(samples.get("split", "unknown"), batch_size)
    inds = _to_int_list(samples.get("ind", -1), batch_size)
    if len(splits) != batch_size:
        splits = [splits[0]] * batch_size
    if len(inds) != batch_size:
        inds = [inds[0]] * batch_size
    return splits, inds


def _attn_dataset_kwargs(
    compression: str,
    chunk_shape: Optional[tuple[int, int, int, int]],
) -> Dict[str, Any]:
    if compression == "none":
        return {}

    kwargs: Dict[str, Any] = {
        "compression": compression,
        "chunks": chunk_shape,
    }
    if compression == "gzip":
        kwargs["compression_opts"] = 4
    return kwargs


def _write_attention_batch(
    attn_ds: h5py.Dataset,
    split_ds: h5py.Dataset,
    ind_ds: h5py.Dataset,
    start_idx: int,
    end_idx: int,
    attn_maps: np.ndarray,
    splits: List[str],
    inds: List[int],
    write_mode: str,
) -> None:
    count = end_idx - start_idx
    if write_mode == "batch":
        attn_ds[start_idx:end_idx] = attn_maps[:count]
        split_ds[start_idx:end_idx] = np.asarray(splits[:count], dtype=object)
        ind_ds[start_idx:end_idx] = np.asarray(inds[:count], dtype=np.int32)
        return

    if write_mode != "sample":
        raise ValueError(f"Unsupported attention write mode: {write_mode}")

    for offset in range(count):
        dst_idx = start_idx + offset
        attn_ds[dst_idx] = attn_maps[offset]
        split_ds[dst_idx] = splits[offset]
        ind_ds[dst_idx] = inds[offset]


def _attention_writer_worker(
    out_path: Path,
    unit: str,
    num_samples: int,
    write_mode: str,
    compression: str,
    pending_writes: "queue.Queue[Optional[tuple[int, int, np.ndarray, List[str], List[int]]]]",
    writer_state: Dict[str, Any],
) -> None:
    writer_state["num_written"] = 0
    writer_state["attn_shape"] = None

    try:
        with h5py.File(out_path, "w") as h5f:
            attn_ds = None
            split_ds = h5f.create_dataset(
                "split",
                shape=(num_samples,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            ind_ds = h5f.create_dataset(
                "ind",
                shape=(num_samples,),
                dtype=np.int32,
            )

            while True:
                item = pending_writes.get()
                try:
                    if item is None:
                        break

                    start_idx, end_idx, attn_maps, splits, inds = item
                    if attn_ds is None:
                        _, n_heads, n_queries, n_tokens = attn_maps.shape
                        chunk_shape = None
                        if compression != "none":
                            chunk_shape = (1, n_heads, n_queries, n_tokens)
                        attn_ds = h5f.create_dataset(
                            "attn_maps",
                            shape=(num_samples, n_heads, n_queries, n_tokens),
                            dtype=np.float16,
                            **_attn_dataset_kwargs(compression, chunk_shape),
                        )
                        h5f.attrs["num_heads"] = int(n_heads)
                        h5f.attrs["num_queries"] = int(n_queries)
                        h5f.attrs["num_memory_tokens"] = int(n_tokens)
                        h5f.attrs["decoder_layers_saved"] = 1
                        writer_state["attn_shape"] = [int(n_heads), int(n_queries), int(n_tokens)]

                    _write_attention_batch(
                        attn_ds=attn_ds,
                        split_ds=split_ds,
                        ind_ds=ind_ds,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        attn_maps=attn_maps,
                        splits=splits,
                        inds=inds,
                        write_mode=write_mode,
                    )
                    writer_state["num_written"] = int(end_idx)
                finally:
                    pending_writes.task_done()

            h5f.attrs["unit"] = unit
            h5f.attrs["num_samples"] = int(writer_state["num_written"])
    except Exception as exc:  # pragma: no cover - defensive cross-thread error path
        writer_state["error"] = exc


def _enqueue_attention_write(
    pending_writes: "queue.Queue[Optional[tuple[int, int, np.ndarray, List[str], List[int]]]]",
    item: tuple[int, int, np.ndarray, List[str], List[int]],
    writer_state: Dict[str, Any],
    writer_thread: threading.Thread,
) -> None:
    while True:
        if writer_state.get("error") is not None:
            raise RuntimeError("Attention writer thread failed.") from writer_state["error"]
        if not writer_thread.is_alive():
            raise RuntimeError("Attention writer thread stopped unexpectedly.")
        try:
            pending_writes.put(item, timeout=0.1)
            return
        except queue.Full:
            continue


def _stop_attention_writer(
    pending_writes: "queue.Queue[Optional[tuple[int, int, np.ndarray, List[str], List[int]]]]",
    writer_state: Dict[str, Any],
    writer_thread: threading.Thread,
) -> None:
    sentinel_enqueued = False
    while writer_thread.is_alive() and writer_state.get("error") is None and not sentinel_enqueued:
        try:
            pending_writes.put(None, timeout=0.1)
            sentinel_enqueued = True
        except queue.Full:
            continue

    if sentinel_enqueued and writer_state.get("error") is None:
        pending_writes.join()
        writer_thread.join()
    else:
        while True:
            try:
                pending_writes.get_nowait()
            except queue.Empty:
                break
            else:
                pending_writes.task_done()
        writer_thread.join(timeout=1)


def _save_unit_attention_maps(
    model: torch.nn.Module,
    data_loader: DataLoader,
    accelerator: Accelerator,
    unit: str,
    out_path: Path,
    write_mode: str,
    compression: str,
    writer_queue_size: int = 2,
) -> Dict[str, Any]:
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = len(data_loader.dataset)
    if num_samples == 0:
        with h5py.File(out_path, "w") as h5f:
            h5f.attrs["unit"] = unit
            h5f.attrs["num_samples"] = 0
        return {
            "unit": unit,
            "path": str(out_path),
            "num_samples": 0,
        }

    write_idx = 0
    writer_state: Dict[str, Any] = {"error": None}
    pending_writes: "queue.Queue[Optional[tuple[int, int, np.ndarray, List[str], List[int]]]]" = (
        queue.Queue(maxsize=writer_queue_size)
    )
    writer_thread = threading.Thread(
        target=_attention_writer_worker,
        kwargs={
            "out_path": out_path,
            "unit": unit,
            "num_samples": num_samples,
            "write_mode": write_mode,
            "compression": compression,
            "pending_writes": pending_writes,
            "writer_state": writer_state,
        },
        name=f"attn-writer-{unit}",
        daemon=True,
    )
    writer_thread.start()

    iterator = tqdm(
        data_loader,
        total=len(data_loader),
        desc=f"Export {unit}",
        leave=False,
        disable=not accelerator.is_main_process,
        unit="batch",
    )

    try:
        with torch.no_grad():
            for samples, _targets in iterator:
                with accelerator.autocast():
                    outputs = model(samples)

                attn_maps = outputs.get("attn_maps")
                if not isinstance(attn_maps, list) or len(attn_maps) == 0:
                    raise RuntimeError(
                        "Model output does not contain decoder attention maps. "
                        "Ensure --attn_maps is enabled and decoder layers are present."
                    )

                # Expected shape for each decoder layer: [B, H, Q, T]
                layer_attn = attn_maps[0]
                if layer_attn is None:
                    raise RuntimeError("Received None attention map for decoder layer 0.")

                layer_attn = np.ascontiguousarray(
                    layer_attn.detach().to(dtype=torch.float16).cpu().numpy()
                )
                if layer_attn.ndim != 4:
                    raise RuntimeError(
                        f"Unexpected attention map rank {layer_attn.ndim}; expected 4 [B, H, Q, T]."
                    )

                bsz = int(layer_attn.shape[0])
                if bsz == 0:
                    continue

                end_idx = min(write_idx + bsz, num_samples)
                count = end_idx - write_idx
                if count <= 0:
                    break

                splits, inds = _extract_batch_metadata(samples, bsz)
                _enqueue_attention_write(
                    pending_writes=pending_writes,
                    item=(
                        write_idx,
                        end_idx,
                        layer_attn[:count],
                        splits[:count],
                        inds[:count],
                    ),
                    writer_state=writer_state,
                    writer_thread=writer_thread,
                )
                write_idx = end_idx

                if writer_state.get("error") is not None:
                    raise RuntimeError("Attention writer thread failed.") from writer_state["error"]

                if accelerator.is_main_process:
                    iterator.set_postfix(saved=write_idx, mode=write_mode)

                if write_idx >= num_samples:
                    break
    finally:
        _stop_attention_writer(
            pending_writes=pending_writes,
            writer_state=writer_state,
            writer_thread=writer_thread,
        )

    if writer_state.get("error") is not None:
        raise RuntimeError("Attention writer thread failed.") from writer_state["error"]

    if write_idx != num_samples or writer_state["num_written"] != num_samples:
        raise RuntimeError(
            f"Attention export wrote {writer_state['num_written']} of {num_samples} samples for unit={unit}."
        )

    return {
        "unit": unit,
        "path": str(out_path),
        "num_samples": int(writer_state["num_written"]),
        "attn_shape_per_sample": writer_state["attn_shape"],
    }


def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()

    # Attention export mode always requests decoder attention maps.
    args.attn_maps = True

    args.backbone_list = BACKBONE_LIST
    set_seed(args.seed)

    if args.resume is None:
        raise ValueError("--resume is required for save_attn_maps.py")

    mp_mode = "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "fp16"
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=mp_mode)
    if accelerator.num_processes != 1:
        raise ValueError(
            "save_attn_maps.py must run with exactly one process. "
            "Launch with: accelerate launch --num_processes 1 save_attn_maps.py ..."
        )

    test_tokens = _normalize_split_spec(args.test_splits)
    eval_units = _expand_split_tokens(test_tokens)
    if len(eval_units) == 0:
        raise ValueError(f"No evaluation units resolved from test_splits={args.test_splits}")

    unit_loaders = {unit: _build_test_loader(args, unit) for unit in eval_units}

    model = NeuroEncoder(args)
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    accelerator.print(
        f"Best checkpoint loaded from {ckpt['epoch']} with best_val_acc={ckpt.get('best_val_acc', 'N/A')}"
    )
    prepared = accelerator.prepare(model, *unit_loaders.values())
    model = prepared[0]

    prepared_unit_loaders = dict(zip(eval_units, prepared[1:]))

    run_name = Path(args.resume).parent.name
    export_root = Path("/engram/nklab/pf2477/multimodal_encoder/attn_maps") / str(args.subj) / run_name
    export_root.mkdir(parents=True, exist_ok=True)

    accelerator.print(f"Using mixed precision: {accelerator.mixed_precision}")
    accelerator.print(f"Exporting attention maps from checkpoint: {args.resume}")
    accelerator.print(f"Output root: {export_root}")
    accelerator.print(
        f"Attention export config: batch_size={args.batch_size}, "
        f"write_mode={args.attn_write_mode}, compression={args.attn_compression}"
    )

    manifest_units: List[Dict[str, Any]] = []
    for unit in eval_units:
        out_file = export_root / f"{unit}.h5py"
        summary = _save_unit_attention_maps(
            model=model,
            data_loader=prepared_unit_loaders[unit],
            accelerator=accelerator,
            unit=unit,
            out_path=out_file,
            write_mode=args.attn_write_mode,
            compression=args.attn_compression,
        )
        manifest_units.append(summary)
        accelerator.print(
            f"Saved unit={unit:>12s} | samples={summary['num_samples']:>6d} | file={out_file}"
        )

    accelerator.print("=" * 60)
    accelerator.print(f"Finished exporting {len(manifest_units)} unit files.")

    if accelerator.is_main_process:
        summary = {
            "checkpoint": args.resume,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "subject": int(args.subj),
            "target_subject": int(args.target_subj),
            "test_splits": args.test_splits,
            "run_name": run_name,
            "output_root": str(export_root),
            "attn_maps_enabled": bool(args.attn_maps),
            "batch_size": int(args.batch_size),
            "attn_write_mode": args.attn_write_mode,
            "attn_compression": args.attn_compression,
            "units": manifest_units,
        }
        out_path = export_root / "manifest.json"
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2)
        accelerator.print(f"Saved attention export manifest to {out_path}")


if __name__ == "__main__":
    main()
