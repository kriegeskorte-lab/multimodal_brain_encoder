from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


@dataclass
class RunningMean:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    def value(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


class RunningPearson:
    """Streaming Pearson correlation across samples for each output dimension.

    Tracks sufficient statistics per output dimension and computes the mean
    Pearson-r across dimensions at finalize time.
    """

    def __init__(self) -> None:
        self.n: int = 0
        self.sum_x: Optional[torch.Tensor] = None
        self.sum_y: Optional[torch.Tensor] = None
        self.sum_x2: Optional[torch.Tensor] = None
        self.sum_y2: Optional[torch.Tensor] = None
        self.sum_xy: Optional[torch.Tensor] = None

    def _ensure_state(self, dim: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.sum_x is not None:
            return
        self.sum_x = torch.zeros(dim, device=device, dtype=dtype)
        self.sum_y = torch.zeros(dim, device=device, dtype=dtype)
        self.sum_x2 = torch.zeros(dim, device=device, dtype=dtype)
        self.sum_y2 = torch.zeros(dim, device=device, dtype=dtype)
        self.sum_xy = torch.zeros(dim, device=device, dtype=dtype)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.ndim > 2:
            pred = pred.reshape(-1, pred.shape[-1])
        if target.ndim > 2:
            target = target.reshape(-1, target.shape[-1])

        pred = pred.detach().to(dtype=torch.float32)
        target = target.detach().to(dtype=torch.float32)

        if pred.numel() == 0:
            return

        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch for Pearson update: pred={pred.shape}, target={target.shape}")

        self._ensure_state(pred.shape[1], pred.device, pred.dtype)
        self.n += int(pred.shape[0])

        self.sum_x.add_(pred.sum(dim=0))
        self.sum_y.add_(target.sum(dim=0))
        self.sum_x2.add_((pred * pred).sum(dim=0))
        self.sum_y2.add_((target * target).sum(dim=0))
        self.sum_xy.add_((pred * target).sum(dim=0))

    def compute(self, eps: float = 1e-8) -> torch.Tensor:
        if self.n == 0 or self.sum_x is None:
            return torch.tensor(0.0)

        n = torch.tensor(float(self.n), device=self.sum_x.device, dtype=self.sum_x.dtype)

        cov = self.sum_xy - (self.sum_x * self.sum_y) / n
        var_x = self.sum_x2 - (self.sum_x * self.sum_x) / n
        var_y = self.sum_y2 - (self.sum_y * self.sum_y) / n

        denom = torch.sqrt(torch.clamp(var_x, min=0.0) * torch.clamp(var_y, min=0.0) + eps)
        corr = cov / denom
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        return corr.mean()

    def finalize(self, accelerator, eps: float = 1e-8) -> float:
        if self.sum_x is None:
            return 0.0

        n_tensor = torch.tensor([float(self.n)], device=self.sum_x.device, dtype=self.sum_x.dtype)
        n_tensor = accelerator.reduce(n_tensor, reduction="sum")

        sum_x = accelerator.reduce(self.sum_x, reduction="sum")
        sum_y = accelerator.reduce(self.sum_y, reduction="sum")
        sum_x2 = accelerator.reduce(self.sum_x2, reduction="sum")
        sum_y2 = accelerator.reduce(self.sum_y2, reduction="sum")
        sum_xy = accelerator.reduce(self.sum_xy, reduction="sum")

        n = n_tensor[0]
        if float(n.item()) <= 0:
            return 0.0
        cov = sum_xy - (sum_x * sum_y) / n
        var_x = sum_x2 - (sum_x * sum_x) / n
        var_y = sum_y2 - (sum_y * sum_y) / n

        denom = torch.sqrt(torch.clamp(var_x, min=0.0) * torch.clamp(var_y, min=0.0) + eps)
        corr = cov / denom
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        return float(corr.mean().item())


def reduce_running_mean(loss_meter: RunningMean, accelerator) -> float:
    stats = torch.tensor([loss_meter.total, float(loss_meter.count)], device=accelerator.device)
    stats = accelerator.reduce(stats, reduction="sum")
    total, count = float(stats[0].item()), int(stats[1].item())
    if count == 0:
        return 0.0
    return total / count
