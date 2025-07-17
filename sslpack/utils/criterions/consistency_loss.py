import torch
import torch.nn.functional as F


def mse_consistency_loss(logits, targets, mask=None, reduction="mean"):
    probs = torch.softmax(logits, dim=-1)
    loss = F.mse_loss(probs, targets, reduction="none")
    if mask is not None:
        loss *= mask
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def ce_consistency_loss(logits, targets, mask=None, reduction="mean"):
    loss = F.cross_entropy(logits, targets, reduction="none")
    if mask is not None:
        loss *= mask
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def kl_consistency_loss(logits, targets, mask=None, reduction="mean"):
    loss = F.kl_div(
        F.log_softmax(logits / 0.5, dim=-1),
        F.softmax(targets / 0.5, dim=-1),
        reduction="none",
    )
    loss = torch.sum(
        loss
        * (1.0 - mask)
        .unsqueeze(dim=-1)
        .repeat(1, torch.softmax(logits, dim=-1).shape[1]),
        dim=1,
    )
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
