import torch


def pair(predictions, targets):
    targets = targets.to(predictions.dtype) if targets.dtype != predictions.dtype else targets
    return predictions.reshape(len(predictions), -1), targets.reshape(len(targets), -1)


def mse(predictions, targets):
    predictions, targets = pair(predictions, targets)
    return ((predictions - targets) ** 2).mean()


def weighted_mse(predictions, targets, weights):
    predictions, targets = pair(predictions, targets)
    scale = torch.as_tensor(weights, dtype=predictions.dtype, device=predictions.device).reshape(1, -1)
    if scale.shape[1] != predictions.shape[1]:
        raise ValueError(f"weighted_mse has {scale.shape[1]} weights for {predictions.shape[1]} target columns")
    return ((predictions - targets) ** 2 * scale).mean()


def mae(predictions, targets):
    predictions, targets = pair(predictions, targets)
    return (predictions - targets).abs().mean()


def huber(predictions, targets, delta=1.0):
    predictions, targets = pair(predictions, targets)
    return torch.nn.functional.huber_loss(predictions, targets, delta=delta)


def log_cosh(predictions, targets):
    predictions, targets = pair(predictions, targets)
    error = predictions - targets
    return (error + torch.nn.functional.softplus(-2.0 * error) - torch.log(torch.tensor(2.0))).mean()
