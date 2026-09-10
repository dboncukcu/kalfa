import torch

from kalfa.registration import lego


def pair(predictions, targets):
    targets = targets.to(predictions.dtype) if targets.dtype != predictions.dtype else targets
    return predictions.reshape(len(predictions), -1), targets.reshape(len(targets), -1)


@lego("/criterion/kalfa/mse", partial=True, alias="mse", description="Mean squared error")
def mse(predictions, targets):
    predictions, targets = pair(predictions, targets)
    return ((predictions - targets) ** 2).mean()


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


@lego("/criterion/kalfa/mae", partial=True, alias="mae", description="Mean absolute error")
def mae(predictions, targets):
    predictions, targets = pair(predictions, targets)
    return (predictions - targets).abs().mean()


@lego("/criterion/kalfa/huber", partial=True, alias="huber",
      description="Huber loss with threshold delta")
def huber(predictions, targets, delta=1.0):
    predictions, targets = pair(predictions, targets)
    return torch.nn.functional.huber_loss(predictions, targets, delta=delta)


@lego("/criterion/kalfa/log_cosh", partial=True, alias="log_cosh",
      description="log(cosh(error)) loss")
def log_cosh(predictions, targets):
    predictions, targets = pair(predictions, targets)
    error = predictions - targets
    return (error + torch.nn.functional.softplus(-2.0 * error) - torch.log(torch.tensor(2.0))).mean()
