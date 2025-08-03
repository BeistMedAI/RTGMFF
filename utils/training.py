from typing import Dict, Tuple, List
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score


from .losses import focal_loss, label_smoothing_loss


def get_loss_function(name: str = 'ce', **kwargs):
    """Return a classification loss function by name.

    Supported names: 'ce' (standard cross entropy), 'focal', 'label_smooth'.
    Additional keyword arguments are forwarded to the underlying loss
    implementation.
    """
    name = name.lower()
    if name == 'ce':
        def _loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(logits, targets, **kwargs)
        return _loss
    elif name == 'focal':
        def _loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return focal_loss(logits, targets, **kwargs)
        return _loss
    elif name in ('label_smooth', 'ls'):  # alias
        def _loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return label_smoothing_loss(logits, targets, **kwargs)
        return _loss
    else:
        raise ValueError(f"Unsupported loss function: {name}")


def compute_losses(logits: torch.Tensor, labels: torch.Tensor,
                   z_proj: torch.Tensor, t_proj: torch.Tensor,
                   alpha: float, beta: float,
                   class_loss_fn=None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute total loss with optional alternative classification loss.

    The classification component is computed by `class_loss_fn` if
    provided; otherwise standard cross entropy is used.  Alignment and
    regularisation terms are added with weights `alpha` and `beta`【569137259789883†screenshot】.
    """
    if class_loss_fn is None:
        class_loss = F.cross_entropy(logits, labels)
    else:
        class_loss = class_loss_fn(logits, labels)
    align_loss = 1.0 - F.cosine_similarity(z_proj, t_proj, dim=-1).mean()
    gram_z = z_proj.T @ z_proj / (z_proj.size(0))
    gram_t = t_proj.T @ t_proj / (t_proj.size(0))
    reg_loss = ((gram_z - gram_t) ** 2).sum()
    total_loss = class_loss + alpha * align_loss + beta * reg_loss
    return total_loss, {
        'loss_ce': class_loss.item(),
        'loss_align': align_loss.item(),
        'loss_reg': reg_loss.item(),
        'loss_total': total_loss.item()
    }


def train_one_epoch(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                    optimiser: torch.optim.Optimizer, device: torch.device,
                    alpha: float, beta: float,
                    class_loss_fn=None) -> Dict[str, float]:
    """Train the model for one epoch.

    The optional `class_loss_fn` can be provided to override the
    default cross entropy.  See `get_loss_function` for helpers to
    construct such functions.
    """
    model.train()
    metrics_accum = {'loss_ce': 0.0, 'loss_align': 0.0, 'loss_reg': 0.0, 'loss_total': 0.0}
    count = 0
    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        ages = batch['age'].to(device)
        genders = batch['gender'].to(device)
        roi_indices = batch['roi_indices'].to(device)
        optimiser.zero_grad()
        logits, aux = model(images, roi_indices, ages, genders)
        loss, metrics = compute_losses(logits, labels,
                                       aux['z_proj'], aux['t_proj'],
                                       alpha=alpha, beta=beta,
                                       class_loss_fn=class_loss_fn)
        loss.backward()
        optimiser.step()
        bs = images.size(0)
        count += bs
        for k in metrics_accum:
            metrics_accum[k] += metrics[k] * bs
    return {k: v / count for k, v in metrics_accum.items()}


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
             device: torch.device) -> Dict[str, float]:
    """Evaluate the model and compute accuracy, sensitivity and specificity."""
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            ages = batch['age'].to(device)
            genders = batch['gender'].to(device)
            roi_indices = batch['roi_indices'].to(device)
            logits, _ = model(images, roi_indices, ages, genders)
            preds = logits.argmax(dim=-1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    if len(set(all_labels)) == 2:
        recall_per_class = recall_score(all_labels, all_preds, average=None, labels=[0, 1])
        specificity = recall_per_class[0]
        sensitivity = recall_per_class[1]
    else:
        sensitivity = recall_score(all_labels, all_preds, average='macro')
        specificity = sensitivity
    return {
        'acc': acc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
