from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from metric import RunningMean, RunningPearson, reduce_running_mean


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion,
    data_loader,
    accelerator,
    target_subj: int,
    split_name: str = "val",
    return_predictions: bool = False,
    return_acc_vector: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, object]:
    model.eval()

    loss_meter = RunningMean()
    pearson_meter = RunningPearson()

    preds_all = [] if return_predictions else None

    total_steps = len(data_loader)
    if max_batches is not None:
        total_steps = min(total_steps, max_batches)

    iterator = tqdm(
        enumerate(data_loader),
        total=total_steps,
        desc=f"Eval {split_name}",
        leave=False,
        disable=not accelerator.is_main_process,
    )

    for step, (samples, targets) in iterator:
        if max_batches is not None and step >= max_batches:
            break

        if isinstance(targets, dict):
            key = f"sub_{target_subj}"
            if key not in targets:
                raise KeyError(f"{key} not found in batch targets: {list(targets.keys())}")
            targets = targets[key].float()
        else:
            targets = targets.float()

        with accelerator.autocast():
            outputs = model(samples)
            pred = outputs["fmri_pred"]
            loss = criterion(outputs, targets)
        batch_size = int(pred.shape[0])
        loss_meter.update(float(loss.detach().item()), batch_size)
        pearson_meter.update(pred, targets)

        if accelerator.is_main_process:
            iterator.set_postfix(loss=f"{loss.detach().item():.4f}")

        if return_predictions:
            gathered_pred = accelerator.gather_for_metrics(pred.detach())
            preds_all.append(gathered_pred.float().cpu().numpy())

    result: Dict[str, object] = {}
    acc_vector = None

    if return_acc_vector:
        split_acc, acc_tensor = pearson_meter.finalize_with_vector(accelerator)
        acc_vector = acc_tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)
    else:
        split_acc = pearson_meter.finalize(accelerator)

    if split_name == "val":
        result["val_loss"] = reduce_running_mean(loss_meter, accelerator)
        result["val_acc"] = split_acc
        if return_acc_vector:
            result["val_acc_vector"] = acc_vector
    elif split_name == "test":
        result["test_loss"] = reduce_running_mean(loss_meter, accelerator)
        result["test_acc"] = split_acc
        if return_acc_vector:
            result["test_acc_vector"] = acc_vector
    else:
        split_loss = reduce_running_mean(loss_meter, accelerator)
        result[f"{split_name}_loss"] = split_loss
        result[f"{split_name}_acc"] = split_acc
        if return_acc_vector:
            result[f"{split_name}_acc_vector"] = acc_vector

    if return_predictions:
        if preds_all:
            result["preds"] = np.concatenate(preds_all, axis=0)
        else:
            result["preds"] = None

    return result
