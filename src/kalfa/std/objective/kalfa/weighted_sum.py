def weighted_sum(models, batch, terms, losses):
    if not isinstance(terms, dict) or not terms:
        raise ValueError("weighted_sum needs terms: a mapping of losses names to weights")
    out = {}
    total = None
    for name, weight in terms.items():
        value = losses[name]
        scalar = value["loss"] if isinstance(value, dict) else value
        out[name] = scalar
        total = float(weight) * scalar if total is None else total + float(weight) * scalar
    out["loss"] = total
    return out
