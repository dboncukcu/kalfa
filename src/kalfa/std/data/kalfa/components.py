import fnmatch

import torch


def weight_of(column, weights):
    for pattern, value in weights.items():
        if pattern != "default" and fnmatch.fnmatchcase(column, str(pattern)):
            return float(value)
    return float(weights.get("default", 1.0))


def class_weights(loader, target=None):
    dataset = loader.dataset
    targets = list(dataset.targets)
    if target is None:
        if len(targets) != 1:
            raise ValueError(f"class_weights needs one target field or a target name; the dataset has {targets}")
        target = targets[0]
    labels = dataset.labels(target).reshape(-1).long()
    counts = torch.bincount(labels).float()
    return counts.sum() / (len(counts) * counts.clamp_min(1.0))


def vocab_size(prep):
    tokenizer = prep.tokenizer() if prep is not None else None
    if tokenizer is None:
        raise ValueError("vocab_size needs a fitted tokenizer among the preprocessors (char_tokenizer)")
    return int(tokenizer.size)


def target_weights(loader, weights, target=None):
    table = loader.dataset.frame.targets
    if target is not None and target not in table:
        raise ValueError(f"target_weights names target {target!r}; the dataset has {sorted(table)}")
    columns = list(table[target]) if target is not None else [column for name in table for column in table[name]]
    return torch.tensor([weight_of(column, dict(weights or {})) for column in columns], dtype=torch.float32)


def feature_width(loader):
    return len(loader.dataset.frame.features)
