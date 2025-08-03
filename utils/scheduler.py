from typing import Optional
import math
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


def cosine_with_warmup(optimizer: optim.Optimizer, total_steps: int,
                       warmup_steps: int = 0, min_lr: float = 0.0) -> _LRScheduler:
    """Create a cosine annealing scheduler with linear warm‑up.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimiser whose learning rate will be scheduled.
    total_steps : int
        Total number of training steps (epochs × batches).
    warmup_steps : int
        Number of warm‑up steps at the start of training.
    min_lr : float
        Minimum learning rate at the end of the schedule.

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler
        Configured scheduler instance.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr, cosine)
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def step_decay(optimizer: optim.Optimizer, step_size: int, gamma: float = 0.1) -> _LRScheduler:
    """Return a StepLR scheduler with given step size and decay factor."""
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)