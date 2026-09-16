def relative_effect(value):
    if not isinstance(value, dict) or len(value) != 1 or next(iter(value)) not in ("times", "plus"):
        return False
    amount = next(iter(value.values()))
    return isinstance(amount, (int, float)) and not isinstance(amount, bool)


def relative(current, value):
    if "times" in value:
        return current * float(value["times"])
    if "plus" in value:
        return current + float(value["plus"])
    raise KeyError(f"a relative effect is {{times: x}} or {{plus: x}}, got {sorted(value)}")


def effect_note(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): effect_note(item) for key, item in value.items()}
    return getattr(value, "__name__", type(value).__name__)


def effective_loss(name, effects, optimizers):
    if f"{name}.loss" in effects:
        return effects[f"{name}.loss"]
    if "loss" in effects and len(optimizers) == 1:
        return effects["loss"]
    if optimizers[name].loss is None:
        raise ValueError(f"optimizer {name!r} names no loss")
    return optimizers[name].loss


def apply_effects(effects, models, optimizers, losses):
    table = dict(losses)
    for key, value in effects.items():
        if "." not in key:
            continue
        owner, param = key.split(".", 1)
        if param == "loss":
            continue
        if owner in models and param == "trainable":
            model = models[owner]
            model.trainable = bool(value)
            for parameter in model.parameters():
                parameter.requires_grad_(bool(value))
        elif owner in optimizers:
            target, _, name = param.rpartition(".")
            optimizers[owner].set_param(name, value, target or None)
        elif owner in table:
            table[owner] = table[owner].with_param(param, value)
        else:
            raise KeyError(f"rule effect {key!r} names neither a model, an optimizer nor a loss")
    return table
