import torch


PREFIX = {"train": "train", "valid": "val", "test": "test"}


def as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach())
    return float(value)


class Means:
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def record(self, name, value, size):
        terms = value if isinstance(value, dict) else {"loss": value}
        for term, item in terms.items():
            if item is None:
                continue
            key = name if term == "loss" else f"{name}/{term}"
            self.sums[key] = self.sums.get(key, 0.0) + as_float(item) * size
            self.counts[key] = self.counts.get(key, 0) + size

    def result(self):
        return {key: total / self.counts[key] for key, total in self.sums.items() if self.counts[key]}


def prefixed(results):
    merged = {}
    for set_name, values in results.items():
        prefix = PREFIX.get(set_name, set_name)
        for name, value in (values or {}).items():
            merged[f"{prefix}/{name}"] = value
    return merged


def history_line(ctx, metrics, seconds):
    line = {"turn": int(ctx.turn), "update": int(ctx.updates)}
    line.update(metrics or {})
    for name, entry in ctx.optimizers.items():
        line[f"lr/{name}"] = entry.rate()
        for label, rate in entry.rates().items():
            line[f"lr/{name}/{label}"] = rate
    for name, entry in ctx.optimizers.items():
        if entry.minimizes is not None:
            line[f"minimizes/{name}"] = entry.minimizes.name
    line["seconds"] = round(float(seconds), 3)
    line["rules"] = list(ctx.fired)
    return line
