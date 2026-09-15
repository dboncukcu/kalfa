import torch


def cross_entropy(predictions, targets, weight=None, label_smoothing=0.0):
    scale = None if weight is None else torch.as_tensor(weight, dtype=predictions.dtype, device=predictions.device)
    logits = predictions.reshape(-1, predictions.shape[-1]) if predictions.ndim > 2 else predictions
    return torch.nn.functional.cross_entropy(logits.float(), targets.reshape(-1).long(), weight=scale,
                                             label_smoothing=label_smoothing)


def bce_logits(predictions, targets, pos_weight=None):
    scale = None if pos_weight is None else torch.as_tensor(pos_weight, dtype=predictions.dtype,
                                                            device=predictions.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        predictions, targets.reshape(predictions.shape).to(predictions.dtype), pos_weight=scale)
