from typing import Optional
import torch
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0,
               alpha: Optional[float] = None, reduction: str = 'mean') -> torch.Tensor:
    """Compute focal loss for classification.

    Parameters
    ----------
    logits : torch.Tensor
        Model outputs of shape (B, C).
    targets : torch.Tensor
        Ground truth labels of shape (B,).
    gamma : float
        Focusing parameter; higher values down‑weight easy examples.
    alpha : Optional[float]
        Class weighting factor for imbalance.  If None no weighting is
        applied.
    reduction : str
        'mean' or 'sum'.
    """
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce
    if alpha is not None:
        alpha_factor = torch.where(targets == 1, alpha, 1 - alpha)
        focal = alpha_factor * focal
    if reduction == 'mean':
        return focal.mean()
    elif reduction == 'sum':
        return focal.sum()
    else:
        return focal


def label_smoothing_loss(logits: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.1,
                         reduction: str = 'mean') -> torch.Tensor:
    """Cross entropy loss with label smoothing.

    Parameters
    ----------
    logits : torch.Tensor
        Model outputs of shape (B, C).
    targets : torch.Tensor
        Ground truth labels of shape (B,).
    smoothing : float
        Smoothing factor; 0 corresponds to standard CE.
    reduction : str
        'mean' or 'sum'.
    """
    num_classes = logits.size(-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1 - smoothing)
    logp = F.log_softmax(logits, dim=-1)
    loss = - (true_dist * logp).sum(dim=-1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
