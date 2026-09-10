import fnmatch

import torch

from kalfa.registration import lego


def weight_of(column, weights):
    for pattern, value in weights.items():
        if pattern != "default" and fnmatch.fnmatchcase(column, str(pattern)):
            return float(value)
    return float(weights.get("default", 1.0))


@lego("/data/kalfa/class_weights", alias="class_weights", counts=True,
      description="Inverse frequency class weights of the train set's target field, mean one; built once the "
                  "train loader exists")
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


@lego("/data/kalfa/vocab_size", alias="vocab_size",
      description="The vocabulary size of the fitted tokenizer among the preprocessors; built once prep exists")
def vocab_size(prep):
    tokenizer = prep.tokenizer() if prep is not None else None
    if tokenizer is None:
        raise ValueError("vocab_size needs a fitted tokenizer among the preprocessors (char_tokenizer)")
    return int(tokenizer.size)


@lego("/data/kalfa/target_weights", alias="target_weights",
      description="A weight per target column, from names and globs resolved against the columns of the target "
                  "field the dataset carries (every target field in order without target), default for the rest; "
                  "built once the train loader exists")
def target_weights(loader, weights, target=None):
    table = loader.dataset.frame.targets
    if target is not None and target not in table:
        raise ValueError(f"target_weights names target {target!r}; the dataset has {sorted(table)}")
    columns = list(table[target]) if target is not None else [column for name in table for column in table[name]]
    return torch.tensor([weight_of(column, dict(weights or {})) for column in columns], dtype=torch.float32)


@lego("/data/kalfa/feature_width", alias="feature_width",
      description="The width of the feature tensor x, from the fitted plan the train loader carries; for a "
                  "layer whose shape follows it (layer_norm)")
def feature_width(loader):
    return len(loader.dataset.frame.features)
