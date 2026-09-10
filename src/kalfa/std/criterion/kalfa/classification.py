import torch

from kalfa.registration import lego


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
