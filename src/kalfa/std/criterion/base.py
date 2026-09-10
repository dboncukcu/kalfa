def pair(predictions, targets):
    targets = targets.to(predictions.dtype) if targets.dtype != predictions.dtype else targets
    return predictions.reshape(len(predictions), -1), targets.reshape(len(targets), -1)
