import fnmatch
import inspect
from dataclasses import dataclass, field

import numpy
import torch

from kalfa.std.common.device import Device
from kalfa.std.pre.base import Prep


def batch_size(batch):
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    raise ValueError("a batch needs at least one tensor field")


def resolve_model(name, models, composites=None, emas=None):
    if name is None:
        raise ValueError("no model name given; write training.predicts")
    if name.endswith(".ema"):
        base = name[:-4]
        if emas and base in emas:
            return emas[base]
        raise KeyError(f"model {base!r} has no ema copy")
    if models and name in models:
        return models[name]
    if composites and name in composites:
        return composites[name]
    known = sorted([*(models or {}), *(composites or {})])
    raise KeyError(f"unknown model {name!r}; the models are {known}")


def call_model(model, batch):
    arguments = []
    for wire in model.inputs:
        if wire not in batch:
            raise KeyError(f"model input {wire!r} is not a batch field; the batch has {sorted(batch)}")
        arguments.append(batch[wire])
    return model(*arguments)


def named_outputs(model, result):
    outputs = list(model.outputs) or ["output"]
    if isinstance(result, (tuple, list)):
        return dict(zip(outputs, result))
    return {outputs[0]: result}


def expand_targets(selector, fields):
    if selector is None:
        return []
    if isinstance(selector, str):
        if any(character in selector for character in "*?["):
            return [name for name in fields if fnmatch.fnmatchcase(name, selector)]
        return [selector]
    return [str(name) for name in selector]


def parameter_names(function):
    try:
        return set(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        return set()


@dataclass
class Pass:
    models: dict
    composites: dict = field(default_factory=dict)
    emas: dict = field(default_factory=dict)
    predicts: str | None = None
    targets: list = field(default_factory=list)
    epoch: int | None = None
    device: Device = field(default_factory=Device.cpu)
    rng: torch.Generator | None = None
    train: bool = False
    prep: Prep | None = None
    set_name: str | None = None
    scaler: torch.amp.GradScaler | None = None
    losses: dict = field(default_factory=dict)
    losses_keys: dict = field(default_factory=dict)
    record: str | None = None
    target_map: dict = field(default_factory=dict)

    def everything(self):
        merged = dict(self.models)
        merged.update(self.composites)
        merged.update({f"{name}.ema": ema for name, ema in self.emas.items()})
        return merged

    def model(self):
        return resolve_model(self.predicts, self.models, self.composites, self.emas)


class Context:
    def __init__(self, batch, scope, step=None):
        self.batch = batch
        self.scope = scope
        self.step = step
        self.computed = None

    @property
    def size(self):
        return batch_size(self.batch)

    def outputs(self):
        if self.computed is None:
            model = self.scope.model()
            self.computed = named_outputs(model, call_model(model, self.batch))
        return self.computed

    def predictions(self, output=None):
        outputs = self.outputs()
        if output is None:
            return next(iter(outputs.values()))
        if output not in outputs:
            raise KeyError(f"model {self.scope.predicts!r} has no output wire {output!r}; it writes {list(outputs)}")
        return outputs[output]

    def selector(self, name=None, output=None):
        if name is not None:
            return name
        if output is not None:
            return self.scope.target_map.get(output)
        if len(self.scope.target_map) == 1:
            return next(iter(self.scope.target_map.values()))
        return None

    def target_fields(self, name=None, output=None):
        selector = self.selector(name, output)
        targets = self.scope.targets
        if selector is None:
            if len(targets) != 1:
                raise ValueError(f"target is not written and the batch has {len(targets)} target fields "
                                 f"{targets}; write target on the definition, or training.targets for the "
                                 f"output wire it names")
            return [targets[0]]
        names = expand_targets(selector, targets)
        if not names:
            raise KeyError(f"target {selector!r} names no target field; the fields are {targets}")
        return names

    def target(self, name=None, output=None):
        if name == "input":
            return self.batch[self.scope.model().inputs[0]]
        names = self.target_fields(name, output)
        missing = [field_name for field_name in names if field_name not in self.batch]
        if missing:
            raise KeyError(f"target fields {missing} are not in the batch; the fields are {sorted(self.batch)}")
        if len(names) == 1:
            return self.batch[names[0]]
        return torch.cat([self.batch[field_name].reshape(len(self.batch[field_name]), -1) for field_name in names],
                         dim=1)

    def rescaled(self, output=None, target=None):
        predictions = self.predictions(output)
        targets = self.target(target, output)
        prep = self.scope.prep
        if prep is None or not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor):
            return predictions, targets
        set_name = self.scope.set_name or "test"
        if target == "input":
            if not prep.rescales():
                return predictions, targets
            return (rescale_tensor(predictions, lambda matrix: prep.rescale_features(matrix, set_name)),
                    rescale_tensor(targets, lambda matrix: prep.rescale_features(matrix, set_name)))
        names = self.target_fields(target, output)
        if not any(prep.rescales(name) for name in names):
            return predictions, targets
        return (rescale_tensor(predictions, lambda matrix: rescale_columns(prep, names, matrix, set_name)),
                rescale_tensor(targets, lambda matrix: rescale_columns(prep, names, matrix, set_name)))

    def detached(self):
        copy = Context(self.batch, self.scope, self.step)
        if self.computed is not None:
            copy.computed = {key: value.detach() if isinstance(value, torch.Tensor) else value
                             for key, value in self.computed.items()}
        return copy


def rescale_columns(prep, names, matrix, set_name):
    out = numpy.array(matrix, dtype="float64")
    if len(names) == 1:
        for position in range(out.shape[1]):
            out[:, position] = prep.rescale(names[0], out[:, position], set_name)
        return out
    for position, name in enumerate(names):
        if position < out.shape[1]:
            out[:, position] = prep.rescale(name, out[:, position], set_name)
    return out


def rescale_tensor(value, transform):
    if not value.is_floating_point():
        return value
    matrix = value.detach().float().cpu().numpy().reshape(len(value), -1)
    rescaled = numpy.asarray(transform(matrix))
    if rescaled.dtype.kind not in "fiu":
        return value
    return torch.from_numpy(rescaled.astype("float32").reshape(tuple(value.shape))).to(value.device)


class Tracker:
    def observe(self, context: Context) -> None:
        raise NotImplementedError

    def record(self, value, size: int) -> None:
        raise NotImplementedError

    def result(self) -> dict:
        raise NotImplementedError


class Loss:
    reads: str = "predictions"

    def loss(self, context: Context, keys: dict | None = None):
        raise NotImplementedError

    def tracker(self, name: str, keys: dict | None = None, rescale: bool = False) -> Tracker:
        raise NotImplementedError

    def with_param(self, name: str, value) -> "Loss":
        raise NotImplementedError

    def resolve(self, **available) -> "Loss":
        return self


class LossView:
    def __init__(self, context):
        self.context = context

    def __contains__(self, name):
        return name in self.context.scope.losses

    def __getitem__(self, name):
        table = self.context.scope.losses
        if name not in table:
            raise KeyError(f"losses has no definition {name!r}; the definitions are {sorted(table)}")
        return table[name].loss(self.context, self.context.scope.losses_keys.get(name))


def entry_active(keys, set_name, turn):
    keys = keys or {}
    sets = keys.get("sets")
    if sets is not None and set_name not in sets:
        return False
    every = int(keys.get("every") or 1)
    return every <= 1 or turn % every == 0


def active_entries(table, keys_table, set_name, turn):
    keys_table = keys_table or {}
    return [(name, entry, keys_table.get(name) or {}) for name, entry in (table or {}).items()
            if entry_active(keys_table.get(name), set_name, turn)]


def resolve_entries(table, **available):
    for entry in (table or {}).values():
        entry.resolve(**available)


def loss_scalar(value):
    if isinstance(value, dict):
        if "loss" not in value:
            raise ValueError(f"an objective returning a mapping must include a 'loss' term, got {sorted(value)}")
        return value["loss"]
    return value


def observe_all(trackers, context):
    with torch.no_grad():
        for tracker in trackers:
            tracker.observe(context)


def collect_results(trackers):
    merged = {}
    for tracker in trackers:
        merged.update(tracker.result())
    return merged


def set_modes(models, train, composites=None):
    for model in models.values():
        if train and model.trainable:
            model.train()
        else:
            model.eval()
    for composite in (composites or {}).values():
        composite.train(bool(train))
