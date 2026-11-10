import copy
import fnmatch
import math

import torch
from torch import nn

from .errors import BuildError, RunError
from .protocols import ContextMetric, Objective


DEFAULT_ON = ("train", "valid", "test")


def as_tuple(on):
    return (on,) if isinstance(on, str) else tuple(on)


def fields(mapping):
    return sorted(str(key) for key in mapping)


def name_of(obj):
    name = getattr(obj, "name", None)
    return name if isinstance(name, str) and name else type(obj).__name__


def unwrap(obj):
    if obj is None:
        return None
    if callable(getattr(obj, "named_parameters", None)):
        return obj
    inner = getattr(obj, "module", None)
    if inner is not None and callable(getattr(inner, "named_parameters", None)):
        return inner
    raise BuildError(f"{type(obj).__name__} {name_of(obj)!r} has no parameters")


def construct(cls, path):
    try:
        return cls()
    except TypeError as problem:
        raise BuildError(f"{path}: {cls.__name__} needs arguments; write it as a call with use") from problem


def grad_norm(parameters):
    grads = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not grads:
        return 0.0
    norms = torch.stack([torch.linalg.vector_norm(grad.float()) for grad in grads])
    return float(torch.linalg.vector_norm(norms))


def frozen_parts(module):
    found = []
    for part in module.modules():
        parameters = list(part.parameters())
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            found.append(part)
    return found


class EmaCopy(nn.Module):
    def __init__(self, module, decay=0.999, every=1):
        super().__init__()
        self.decay = float(decay)
        self.every = int(every)
        self.module = copy.deepcopy(module)
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        nn.Module.train(self, False)
        self.updates = 0
        object.__setattr__(self, "source", module)

    def shift(self, module):
        self.updates += 1
        if self.every > 1 and self.updates % self.every:
            return
        module = unwrap(module)
        with torch.no_grad():
            own = dict(self.module.named_parameters())
            for name, parameter in module.named_parameters():
                kept = own.get(name)
                if kept is None:
                    raise RunError(f"the EMA copy has no parameter {name!r}; the module changed after it was copied")
                kept.mul_(self.decay).add_(parameter.detach().to(kept.device), alpha=1.0 - self.decay)
            buffers = dict(self.module.named_buffers())
            for name, buffer in module.named_buffers():
                kept = buffers.get(name)
                if kept is not None:
                    kept.copy_(buffer)

    def forward(self, *inputs):
        return self.module(*inputs)

    def train(self, mode=True):
        return self

    def get_extra_state(self):
        return {"updates": self.updates}

    def set_extra_state(self, state):
        self.updates = int((state or {}).get("updates", 0))


class ModelEntry:
    def __init__(self, name, module, inputs, outputs, ema=None, freeze=False, weights=None, init=None):
        self.name = name
        self.module = module
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.ema = ema
        self.freeze = freeze
        self.weights = weights
        self.init = init
        self.summary = None
        self.path = f"models.{name}"
        self.locked = False
        self.frozen_parts = []
        if freeze:
            self.set_freeze(freeze)

    def call(self, batch):
        inputs = []
        for name in self.inputs:
            if name not in batch:
                raise RunError(f"model {self.name!r} needs the batch field {name!r}; the batch has {fields(batch)}")
            inputs.append(batch[name])
        return self.name_outputs(self.module(*inputs))

    def name_outputs(self, result):
        names = self.outputs or [self.name]
        if isinstance(result, torch.Tensor):
            return {names[0]: result}
        if isinstance(result, dict):
            missing = [name for name in names if name not in result]
            if missing:
                raise RunError(f"model {self.name!r} returned a dict without {missing}; its keys are {fields(result)}")
            return result
        if isinstance(result, (tuple, list)):
            if len(result) != len(names):
                raise RunError(f"model {self.name!r} returned {len(result)} values for the outputs {names}")
            return dict(zip(names, result))
        raise RunError(f"model {self.name!r} returned {type(result).__name__}, not a tensor, a tuple or a dict")

    def set_freeze(self, value):
        freeze_parameters(self.module, value)
        self.freeze = value
        self.locked = value is True
        self.frozen_parts = [] if self.locked or not value else frozen_parts(self.module)
        if self.locked:
            self.module.eval()

    @property
    def frozen(self):
        return self.freeze

    def set_mode(self, train):
        self.module.train(bool(train) and not self.locked)
        for part in self.frozen_parts:
            part.eval()
        return self

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.module.parameters())

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        self.module.load_state_dict(state)


def align(guess, truth):
    if isinstance(guess, torch.Tensor) and isinstance(truth, torch.Tensor):
        if truth.ndim == guess.ndim - 1 and guess.ndim > 0 and guess.shape[-1] == 1:
            guess = guess.squeeze(-1)
        elif guess.ndim == truth.ndim - 1 and truth.ndim > 0 and truth.shape[-1] == 1:
            truth = truth.squeeze(-1)
    return guess, truth


def pair_values(owner, outputs, batch, ctx, kind):
    label = f"{kind} {owner.name!r}" if getattr(owner, "name", "") else kind
    if owner.pred is None:
        raise RunError(f"{label}: pred is not set")
    if owner.target is None:
        raise RunError(f"{label}: target is not set")
    outputs = {} if outputs is None else outputs
    if owner.pred not in outputs:
        raise RunError(f"{label}: output {owner.pred!r} is not among the step outputs {fields(outputs)}")
    if owner.target not in batch:
        raise RunError(f"{label}: field {owner.target!r} is not in the batch; its fields are {fields(batch)}")
    guess, truth = align(outputs[owner.pred], batch[owner.target])
    if owner.rescale:
        inverse = getattr(getattr(ctx, "data", None), "inverse", None)
        if callable(inverse):
            guess = inverse(owner.target, guess)
            truth = inverse(owner.target, truth)
    return guess, truth


def apply_keys(obj, keys, path):
    for key in ("pred", "target", "rescale"):
        if key not in keys:
            continue
        if not hasattr(obj, key):
            raise BuildError(f"{path}: {key} belongs to a criterion or a metric, not to {type(obj).__name__}")
        value = keys[key]
        setattr(obj, key, bool(value) if key == "rescale" and value is not None else value)
    if "on" in keys:
        obj.on = as_tuple(keys["on"])
    if "every" in keys:
        obj.every = int(keys["every"])


class Criterion(Objective):
    def __init__(self, function, pred=None, target=None, rescale=False, on=("train", "valid", "test"), every=1):
        super().__init__()
        self.function = function
        self.pred = pred
        self.target = target
        self.rescale = bool(rescale)
        self.on = (on,) if isinstance(on, str) else tuple(on)
        self.every = int(every)

    def pair(self, outputs, batch, ctx):
        return pair_values(self, outputs, batch, ctx, "loss")

    def forward(self, outputs, batch, ctx):
        guess, truth = self.pair(outputs, batch, ctx)
        return self.function(guess, truth)


def wrap_loss(obj, keys, name, path):
    keys = dict(keys or {})
    if isinstance(obj, type):
        obj = construct(obj, path)
    if isinstance(obj, Objective):
        apply_keys(obj, keys, path)
        result = obj
    elif callable(obj):
        result = Criterion(obj, pred=keys.get("pred"), target=keys.get("target"), rescale=bool(keys.get("rescale")),
                           on=keys.get("on", DEFAULT_ON), every=keys.get("every", 1))
    else:
        raise BuildError(f"{path}: {type(obj).__name__} is not a loss; a loss is an Objective or a callable "
                         "taking (pred, target)")
    result.name = name
    result.path = path
    return result


def as_float(value):
    if isinstance(value, dict):
        return {str(key): as_float(inner) for key, inner in value.items()}
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return {str(index): float(item) for index, item in enumerate(value.detach().flatten().tolist())}
    if isinstance(value, (list, tuple)):
        return {str(index): as_float(item) for index, item in enumerate(value)}
    return float(value)


class MetricEntry:
    def __init__(self, name, metric, pred=None, target=None, rescale=None, on=("train", "valid", "test"), every=1):
        self.name = name
        self.metric = metric
        self.pred = pred
        self.target = target
        self.rescale = rescale
        self.on = (on,) if isinstance(on, str) else tuple(on)
        self.every = int(every)
        self.path = f"metrics.{name}"
        self.placed = False
        self.seen = 0

    def reset(self):
        self.metric.reset()
        self.seen = 0

    def update(self, outputs, batch, ctx):
        if isinstance(self.metric, ContextMetric):
            self.metric.update(outputs, batch, ctx)
            self.seen += 1
            return
        guess, truth = pair_values(self, outputs, batch, ctx, "metric")
        if isinstance(guess, torch.Tensor):
            guess = guess.detach()
            if not self.placed:
                self.to(guess.device)
        if isinstance(truth, torch.Tensor):
            truth = truth.detach()
            if not truth.is_floating_point() and not truth.is_complex():
                truth = truth.long()
        self.metric.update(guess, truth)
        self.seen += 1

    def compute(self):
        if not self.seen:
            return math.nan
        return as_float(self.metric.compute())

    def to(self, device):
        mover = getattr(self.metric, "to", None)
        if callable(mover):
            mover(device)
        self.placed = True
        return self

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


def wrap_metric(obj, keys, name, path):
    keys = dict(keys or {})
    if isinstance(obj, type):
        obj = construct(obj, path)
    if isinstance(obj, ContextMetric):
        apply_keys(obj, keys, path)
        obj.name = name
        obj.path = path
        return obj
    if not all(callable(getattr(obj, method, None)) for method in ("update", "compute", "reset")):
        raise BuildError(f"{path}: {type(obj).__name__} is not a metric; a metric has update(pred, target), "
                         "compute() and reset(), or it is a ContextMetric")
    entry = MetricEntry(name, obj, pred=keys.get("pred"), target=keys.get("target"), rescale=keys.get("rescale"),
                        on=keys.get("on", DEFAULT_ON), every=keys.get("every", 1))
    entry.path = path
    return entry


class Group:
    def __init__(self, of, match=None, **params):
        self.of = of
        self.match = match
        self.params = dict(params)


class OptimizerEntry:
    def __init__(self, name, target, params, over, minimizes=None, schedule=None):
        self.name = name
        self.target = target
        self.params = dict(params)
        self.over = list(over)
        self.minimizes = minimizes
        self.schedule = schedule
        self.real = None
        self.base = None
        self.scaler = None
        self.updates = 0
        self.pending = None
        self.emas = []
        self.origins = []
        self.path = f"optimizers.{name}"

    def bind(self, ctx):
        self.build()

    def build(self):
        if self.real is not None:
            return self.real
        claimed = set()
        groups = []
        origins = []
        for position, item in enumerate(self.over):
            if not isinstance(item, Group):
                continue
            module = unwrap(item.of)
            if module is None:
                continue
            picked = [parameter for name, parameter in module.named_parameters()
                      if (item.match is None or fnmatch.fnmatchcase(name, item.match)) and id(parameter) not in claimed]
            if not picked:
                wanted = f" matching {item.match!r}" if item.match else ""
                raise BuildError(f"{self.path}.over[{position}]: no parameter of {name_of(module)} is left for the "
                                 f"group{wanted}")
            claimed.update(id(parameter) for parameter in picked)
            groups.append({"params": picked, **item.params})
            origins.append(item)
        rest = []
        for item in self.over:
            if isinstance(item, Group):
                continue
            module = unwrap(item)
            if module is None:
                continue
            for parameter in module.parameters():
                if id(parameter) not in claimed:
                    claimed.add(id(parameter))
                    rest.append(parameter)
        if rest:
            groups.insert(0, {"params": rest})
            origins.insert(0, None)
        if not groups:
            raise BuildError(f"{self.path} has no parameters to optimize")
        self.real = self.target(groups, **self.params)
        self.origins = origins
        self.base = [float(group.get("lr", math.nan)) for group in self.real.param_groups]
        if self.pending is not None:
            self.real.load_state_dict(self.pending)
            self.pending = None
            self.restore_params()
        self.reschedule()
        return self.real

    def restore_params(self):
        for index, (group, origin) in enumerate(zip(self.real.param_groups, self.origins)):
            for name, value in self.params.items():
                if origin is None or name not in origin.params:
                    group[name] = value
            if origin is not None:
                group.update(origin.params)
            self.base[index] = float(group.get("lr", math.nan))

    def owned(self):
        self.build()
        found = []
        seen = set()
        for group in self.real.param_groups:
            for parameter in group["params"]:
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    found.append(parameter)
        return found

    def modules(self):
        found = []
        for item in self.over:
            module = unwrap(item.of if isinstance(item, Group) else item)
            if module is not None and all(module is not other for other in found):
                found.append(module)
        return found

    def parameters(self):
        found = []
        seen = set()
        for module in self.modules():
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    found.append(parameter)
        return found

    def scaling(self):
        if self.scaler is None:
            return False
        enabled = getattr(self.scaler, "is_enabled", None)
        return bool(enabled()) if callable(enabled) else True

    def reschedule(self):
        if self.schedule is None or self.real is None:
            return
        factor = float(self.schedule(self.updates))
        for group, base in zip(self.real.param_groups, self.base):
            group["lr"] = base * factor

    def zero_grad(self):
        for parameter in self.owned():
            parameter.grad = None

    def backward(self, loss):
        if self.scaling():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip(self, limit):
        parameters = self.owned()
        if self.scaling():
            self.scaler.unscale_(self.real)
        if limit is None:
            return grad_norm(parameters)
        return float(torch.nn.utils.clip_grad_norm_(parameters, float(limit)))

    def step(self):
        self.build()
        if self.scaling():
            self.scaler.step(self.real)
            self.scaler.update()
        else:
            self.real.step()
        self.updates += 1
        self.reschedule()
        self.shift_emas()

    def shift_emas(self):
        for ema in self.emas:
            ema.shift(ema.source)

    def minimize(self, loss, clip=None):
        self.zero_grad()
        self.backward(loss)
        norm = self.clip(clip)
        self.step()
        return norm

    def rate(self):
        if self.real is None:
            return float(self.params.get("lr", math.nan))
        return float(self.real.param_groups[0]["lr"])

    def group_label(self, group):
        label = getattr(group, "label", None) or name_of(unwrap(group.of) or group.of)
        return f"{label}[{group.match}]" if group.match else label

    def rates(self):
        found = {}
        if self.real is None:
            for item in self.over:
                if isinstance(item, Group):
                    found[self.group_label(item)] = float(item.params.get("lr", self.params.get("lr", math.nan)))
            return found
        for index, origin in enumerate(self.origins):
            if origin is not None:
                found[self.group_label(origin)] = float(self.real.param_groups[index]["lr"])
        return found

    def set_param(self, name, value):
        if name in ("minimizes", "schedule", "over"):
            setattr(self, name, value)
            if name == "schedule":
                self.reschedule()
            return
        self.params[name] = value
        if self.real is None:
            return
        for index, (group, origin) in enumerate(zip(self.real.param_groups, self.origins)):
            if origin is not None and name in origin.params:
                continue
            group[name] = value
            if name == "lr":
                self.base[index] = float(value)
        self.reschedule()

    def state_dict(self):
        inner = self.pending if self.real is None else self.real.state_dict()
        state = {"torch": inner, "updates": self.updates, "params": dict(self.params),
                 "base": None if self.base is None else list(self.base)}
        saver = getattr(self.scaler, "state_dict", None)
        if self.scaling() and callable(saver):
            state["scaler"] = saver()
        return state

    def load_state_dict(self, state):
        if not state:
            return
        self.updates = int(state.get("updates", 0))
        loader = getattr(self.scaler, "load_state_dict", None)
        if state.get("scaler") is not None and callable(loader):
            loader(state["scaler"])
        inner = state.get("torch")
        if inner is None:
            return
        if self.real is None:
            self.pending = inner
            return
        self.real.load_state_dict(inner)
        self.restore_params()
        self.reschedule()


def freeze_parameters(module, spec):
    module = unwrap(module)
    if spec is True:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        return
    patterns = [] if not spec else ([spec] if isinstance(spec, str) else [str(pattern) for pattern in spec])
    for name, parameter in module.named_parameters():
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
            parameter.requires_grad_(False)
        elif parameter.is_floating_point() or parameter.is_complex():
            parameter.requires_grad_(True)


def active(entry, set_name, turn):
    on = getattr(entry, "on", None)
    if on is not None and set_name not in on:
        return False
    every = int(getattr(entry, "every", 1) or 1)
    return every <= 1 or turn % every == 0


def loss_scalar(value, name):
    if isinstance(value, dict):
        if "loss" not in value:
            raise ValueError(f"loss {name!r} returned a mapping without a 'loss' key; its keys are {sorted(value)}")
        return value["loss"]
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"loss {name!r} returned {type(value).__name__}, not a tensor")
    return value
