# loss.py
import torch
import torch.nn as nn


def weightedHuberLoss(pred, target,beta=0.4,alpha=0.8):
    """
    pred, target: [B, 1]
    """
    error = pred - target
    abs_error = torch.abs(error)

    huber = torch.where(
        abs_error < beta,
        0.5 * error ** 2 / beta,
        abs_error * beta - 0.5 * beta
    )

    # weights = 1.0 + alpha * (target+0.8)
    weights = 1.0 + alpha * torch.log1p(target + 0.8)

    return (weights * huber).mean()

