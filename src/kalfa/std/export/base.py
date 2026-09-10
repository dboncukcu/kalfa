from pathlib import Path

import torch


def traced_inputs(model, batch, rows=2):
    inputs = []
    for wire in model.inputs:
        if wire not in batch:
            raise KeyError(f"model input {wire!r} is not a batch field; the batch has {sorted(batch)}")
        value = batch[wire]
        inputs.append(value[:rows] if isinstance(value, torch.Tensor) else value)
    return tuple(inputs)


def target_path(directory, stem, suffix):
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{stem}.{suffix}"
