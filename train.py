from __future__ import annotations

from typing import Dict, Optional

import torch
from tqdm.auto import tqdm

from metric import RunningMean, RunningPearson, reduce_running_mean


def train_one_epoch(
    model: torch.nn.Module,
    criterion,
    data_loader,
    optimizer: torch.optim.Optimizer,
    accelerator,
    epoch: int,
    target_subj: int,
    max_grad_norm: float = 0.0,
    max_batches: Optional[int] = None,
    log_interval: int = 20,
) -> Dict[str, float]:
    model.train()

    loss_meter = RunningMean()
    pearson_meter = RunningPearson()

    total_steps = len(data_loader)
    if max_batches is not None:
        total_steps = min(total_steps, max_batches)

    iterator = tqdm(
        enumerate(data_loader),
        total=total_steps,
        desc=f"Train {epoch:03d}",
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

        outputs = model(samples)
        loss = criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)

        if max_grad_norm > 0.0:
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        pred = outputs["fmri_pred"]
        batch_size = int(pred.shape[0])
        loss_meter.update(float(loss.detach().item()), batch_size)
        pearson_meter.update(pred, targets)

        if accelerator.is_main_process:
            iterator.set_postfix(loss=f"{loss.detach().item():.4f}")

        # if (step + 1) % log_interval == 0:
        #     accelerator.print(
        #         f"Epoch {epoch:03d} | step {step + 1:05d} | "
        #         f"loss={loss.detach().item():.6f}"
        #     )

    train_loss = reduce_running_mean(loss_meter, accelerator)
    train_acc = pearson_meter.finalize(accelerator)

    lr = 0.0
    if len(optimizer.param_groups) > 0:
        lr = float(optimizer.param_groups[0].get("lr", 0.0))

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "lr": lr,
    }
