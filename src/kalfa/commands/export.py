import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from ..core import run as running
from ..core.errors import KalfaError
from ..core.record import atomic
from .common import loaded_which


FORMATS = ("state_dict", "onnx")
SUFFIXES = {"state_dict": ".pt", "onnx": ".onnx"}
SETS = ("test", "valid", "train")
LEGACY = "You are using the legacy TorchScript-based ONNX export"
REMOVED = "The feature will be removed"


@dataclass
class Exported:
    record: str
    path: str
    format: str
    model: str
    which: str


def need_onnx():
    try:
        import onnx
    except ImportError as exc:
        raise KalfaError("onnx export needs the onnx package (uv sync installs the dependencies of kalfa)") from exc
    return onnx


def chosen_entry(ctx, record, model):
    if model is not None:
        return running.model_entry(ctx, model)
    entry = ctx.predicts()
    if entry is None:
        raise KalfaError(f"{record}: the run has the models {sorted(ctx.models)} and no train.predicts; name one with "
                         f"model")
    return entry


def first_batch(ctx, record):
    loaders = ctx.loaders()
    for name in SETS:
        loader = loaders.get(name)
        if loader is None:
            continue
        batch = next(iter(loader), None)
        if batch is not None:
            return name, batch
    raise KalfaError(f"{record}: onnx export traces the model on a batch, and the run has no test, valid or train "
                     f"batch; are the data files still where the run read them?")


def save_state(entry, path):
    state = entry.module.state_dict()
    for key, value in list(state.items()):
        if isinstance(value, torch.Tensor):
            state[key] = value.detach().cpu()
    with atomic(path) as temporary:
        torch.save(state, temporary)


def save_onnx(ctx, entry, record, path, opset, dynamic):
    name, batch = first_batch(ctx, record)
    if ctx.device is not None:
        batch = ctx.device.move(batch)
    fields = batch if isinstance(batch, dict) else {}
    missing = [field for field in entry.inputs if field not in fields]
    if missing:
        raise KalfaError(f"{record}: model {entry.name!r} takes {missing}, which the first {name} batch does not "
                         f"have; its fields are {sorted(fields)}")
    inputs = tuple(fields[field] for field in entry.inputs)
    input_names = list(entry.inputs)
    output_names = list(entry.outputs or [entry.name])
    axes = {field: {0: "batch"} for field in input_names + output_names} if dynamic else None
    entry.module.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=LEGACY, category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=REMOVED, category=DeprecationWarning, module=r"torch\.onnx")
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning, module=r"torch\.(?!jit)")
        torch.onnx.export(entry.module, inputs, str(path), opset_version=opset, input_names=input_names,
                          output_names=output_names, dynamic_axes=axes, dynamo=False)


def target_of(source, out, name):
    if not out:
        return source / "exports" / name
    target = Path(out)
    if target.is_dir() or str(out).endswith(("/", os.sep)):
        return target / name
    return target


def export_record(record, format="state_dict", model=None, which=None, out=None, device=None, opset=17,
                  dynamic=True):
    if format not in FORMATS:
        raise KalfaError(f"format must be one of {list(FORMATS)}, got {format!r}")
    if format == "onnx":
        need_onnx()
    ctx = running.open_record(record, which=which, device=device, data=format == "onnx")
    source = Path(record)
    entry = chosen_entry(ctx, record, model)
    chosen = loaded_which(ctx, source, which)
    path = target_of(source, out, f"{entry.name}_{chosen}{SUFFIXES[format]}")
    if format == "state_dict":
        save_state(entry, path)
    else:
        save_onnx(ctx, entry, record, path, int(opset), dynamic)
    return Exported(str(record), str(path), format, entry.name, chosen)
