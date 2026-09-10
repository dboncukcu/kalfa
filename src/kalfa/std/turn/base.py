import functools


def effective_loss(name, effects, optimizers):
    """The loss an optimizer minimizes this turn: the rule effects win over the optimizer's own loss."""
    if f"{name}.loss" in effects:
        return effects[f"{name}.loss"]
    if "loss" in effects and len(optimizers) == 1:
        return effects["loss"]
    loss = getattr(optimizers[name], "loss", None)
    if loss is None:
        raise ValueError(f"optimizer {name!r} names no loss")
    return loss


def with_param(entry, param, value):
    """A losses entry with one param replaced: adapters know how, an objective is a partial rebuilt."""
    if hasattr(entry, "with_param"):
        return entry.with_param(param, value)
    base = getattr(entry, "func", entry)
    kwargs = dict(getattr(entry, "keywords", None) or {})
    kwargs[param] = value
    return functools.partial(base, **kwargs)


def apply_effects(effects, models, optimizers, losses):
    """Apply the rule effects that change objects: loss params, optimizer params, trainable flags.

    Returns the losses table with per param replacements applied.
    """
    table = dict(losses)
    for key, value in effects.items():
        if "." not in key:
            continue
        owner, param = key.split(".", 1)
        if param == "loss":
            continue
        if owner in models and param == "trainable":
            model = models[owner]
            model.kalfa_trainable = bool(value)
            for parameter in model.parameters():
                parameter.requires_grad_(bool(value))
        elif owner in optimizers:
            optimizers[owner].set_param(param, value)
        elif owner in table:
            table[owner] = with_param(table[owner], param, value)
        else:
            raise KeyError(f"rule effect {key!r} names neither a model, an optimizer nor a loss")
    return table
