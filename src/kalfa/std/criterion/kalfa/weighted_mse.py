import torch

from kalfa.registration import lego
from kalfa.std.criterion.base import pair


@lego("/criterion/kalfa/weighted_mse", partial=True, alias="weighted_mse", refs={"weights": "data"},
      description="Mean squared error with a weight per target column; weights is a list in column order or "
                  "{uri: target_weights, params: {weights: {column: 3.0, 'glob*': 1.5, default: 1.0}}}, named "
                  "against the target columns the dataset carries")
def weighted_mse(predictions, targets, weights):
    predictions, targets = pair(predictions, targets)
    scale = torch.as_tensor(weights, dtype=predictions.dtype, device=predictions.device).reshape(1, -1)
    if scale.shape[1] != predictions.shape[1]:
        raise ValueError(f"weighted_mse has {scale.shape[1]} weights for {predictions.shape[1]} target columns")
    return ((predictions - targets) ** 2 * scale).mean()
