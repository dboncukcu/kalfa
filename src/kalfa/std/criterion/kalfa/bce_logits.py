import torch

from kalfa.registration import lego


@lego("/criterion/kalfa/bce_logits", partial=True, alias="bce_logits",
      description="Binary cross entropy on logits")
def bce_logits(predictions, targets, pos_weight=None):
    scale = None if pos_weight is None else torch.as_tensor(pos_weight, dtype=predictions.dtype,
                                                            device=predictions.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        predictions, targets.reshape(predictions.shape).to(predictions.dtype), pos_weight=scale)
