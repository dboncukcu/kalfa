"""Criteria: differentiable (predictions, targets) legos used under losses and metrics."""

import torch

from ..registration import lego


def _pair(predictions, targets):
    targets = targets.to(predictions.dtype) if targets.dtype != predictions.dtype else targets
    return predictions.reshape(len(predictions), -1), targets.reshape(len(targets), -1)


@lego("/criterion/kalfa/mse", partial=True, alias="mse", description="Mean squared error")
def mse(predictions, targets):
    predictions, targets = _pair(predictions, targets)
    return ((predictions - targets) ** 2).mean()


@lego("/criterion/kalfa/mae", partial=True, alias="mae", description="Mean absolute error")
def mae(predictions, targets):
    predictions, targets = _pair(predictions, targets)
    return (predictions - targets).abs().mean()


@lego("/criterion/kalfa/huber", partial=True, alias="huber",
            description="Huber loss with threshold delta")
def huber(predictions, targets, delta=1.0):
    predictions, targets = _pair(predictions, targets)
    return torch.nn.functional.huber_loss(predictions, targets, delta=delta)


@lego("/criterion/kalfa/log_cosh", partial=True, alias="log_cosh",
            description="log(cosh(error)) loss")
def log_cosh(predictions, targets):
    predictions, targets = _pair(predictions, targets)
    error = predictions - targets
    return (error + torch.nn.functional.softplus(-2.0 * error) - torch.log(torch.tensor(2.0))).mean()


@lego("/criterion/kalfa/cross_entropy", partial=True, alias="cross_entropy",
            description="Cross entropy over class logits; weight may be a run time component")
def cross_entropy(predictions, targets, weight=None, label_smoothing=0.0):
    scale = None if weight is None else torch.as_tensor(weight, dtype=predictions.dtype, device=predictions.device)
    logits = predictions.reshape(-1, predictions.shape[-1]) if predictions.ndim > 2 else predictions
    return torch.nn.functional.cross_entropy(logits.float(), targets.reshape(-1).long(), weight=scale,
                                             label_smoothing=label_smoothing)


@lego("/criterion/kalfa/bce_logits", partial=True, alias="bce_logits",
            description="Binary cross entropy on logits")
def bce_logits(predictions, targets, pos_weight=None):
    scale = None if pos_weight is None else torch.as_tensor(pos_weight, dtype=predictions.dtype,
                                                            device=predictions.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        predictions, targets.reshape(predictions.shape).to(predictions.dtype), pos_weight=scale)
