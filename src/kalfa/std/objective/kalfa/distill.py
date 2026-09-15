import torch

from kalfa.std.objective.base import input_of


def target_of(batch, model, target=None):
    if target is not None:
        return batch[target]
    wires = set(model.inputs)
    rest = [name for name in batch if name not in wires]
    if len(rest) != 1:
        raise ValueError(f"the objective cannot tell the target field among {rest}; write target")
    return batch[rest[0]]


def distill(models, batch, student, teacher, temperature=1.0, alpha=0.5, target=None):
    learner = models[student]
    x = input_of(learner, batch)
    labels = target_of(batch, learner, target).reshape(-1).long()
    logits = learner(x)
    with torch.no_grad():
        guide = models[teacher](x)
    scale = float(temperature)
    log_soft = torch.log_softmax(logits / scale, dim=-1)
    soft = torch.softmax(guide / scale, dim=-1)
    kl = torch.nn.functional.kl_div(log_soft, soft, reduction="batchmean") * scale * scale
    ce = torch.nn.functional.cross_entropy(logits, labels)
    return {"loss": float(alpha) * kl + (1.0 - float(alpha)) * ce, "ce": ce, "kl": kl}
