"""Run time support shared by the std legos: batch handling, model lookup and the prediction context."""

import math

import numpy
import torch


def to_device(batch, device):
    if device is None:
        return batch
    target = torch.device(device)
    return {key: value.to(target) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def batch_size(batch):
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    raise ValueError("a batch needs at least one tensor field")


def resolve_model(name, models, composites=None, emas=None):
    """The module behind a model name: a trained model, a composite, or ``<name>.ema``."""
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


def model_inputs(model):
    inputs = getattr(model, "inputs", None)
    if inputs is None:
        raise TypeError(f"{type(model).__name__} does not name its input wires")
    return list(inputs)


def model_outputs(model):
    outputs = getattr(model, "outputs", None)
    if outputs is None:
        return ["output"]
    return list(outputs)


def call_model(model, batch):
    """Call a model on a batch, binding its input wires to batch fields by name."""
    args = []
    for wire in model_inputs(model):
        if wire not in batch:
            raise KeyError(f"model input {wire!r} is not a batch field; the batch has {sorted(batch)}")
        args.append(batch[wire])
    return model(*args)


def named_outputs(model, result):
    outputs = model_outputs(model)
    if isinstance(result, (tuple, list)):
        return dict(zip(outputs, result))
    return {outputs[0]: result}


class Context:
    """One batch as the adapters see it: the predictions of the ``predicts`` model, computed once, and the targets."""

    def __init__(self, batch, models, composites=None, emas=None, predicts=None, targets=(), step=None, epoch=None,
                 rng=None, train=False, prep=None, set_name=None, scaler=None, losses=None, losses_keys=None,
                 record=None):
        self.batch = batch
        self.losses = losses
        self.losses_keys = losses_keys
        self.record = record
        self.models = models
        self.composites = composites or {}
        self.emas = emas or {}
        self.predicts = predicts
        self.targets = list(targets)
        self.step = step
        self.epoch = epoch
        self.rng = rng
        self.train = train
        self.prep = prep
        self.set_name = set_name
        self.scaler = scaler
        self._outputs = None

    def everything(self):
        """Every model by name: trained models, composites and the ema copies as ``<name>.ema``."""
        merged = dict(self.models)
        merged.update(self.composites)
        merged.update({f"{name}.ema": ema for name, ema in self.emas.items()})
        return merged

    @property
    def size(self):
        return batch_size(self.batch)

    def model(self):
        return resolve_model(self.predicts, self.models, self.composites, self.emas)

    def outputs(self):
        if self._outputs is None:
            model = self.model()
            self._outputs = named_outputs(model, call_model(model, self.batch))
        return self._outputs

    def predictions(self, output=None):
        outputs = self.outputs()
        if output is None:
            return next(iter(outputs.values()))
        if output not in outputs:
            raise KeyError(f"model {self.predicts!r} has no output wire {output!r}; it writes {list(outputs)}")
        return outputs[output]

    def target(self, name=None):
        if name == "input":
            wires = model_inputs(self.model())
            return self.batch[wires[0]]
        if name is None:
            if len(self.targets) != 1:
                raise ValueError(f"target is not written and the batch has {len(self.targets)} target fields "
                                 f"{self.targets}; write target on the definition")
            name = self.targets[0]
        if name not in self.batch:
            raise KeyError(f"target field {name!r} is not in the batch; the fields are {sorted(self.batch)}")
        return self.batch[name]

    def rescaled(self, output=None, target=None):
        """Predictions and target in the original scale: the rescaling preprocessors of the target field undone.

        Without a prep, or for a target whose chain rescales nothing, the tensors come back as they are.
        """
        predictions = self.predictions(output)
        targets = self.target(target)
        if self.prep is None or not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor):
            return predictions, targets
        set_name = self.set_name or "test"
        if target == "input":
            if not self.prep.rescales():
                return predictions, targets
            transform = lambda matrix: self.prep.rescale_features(matrix, set_name)  # noqa: E731
        else:
            name = target if target is not None else (self.targets[0] if len(self.targets) == 1 else None)
            if name is None or not self.prep.rescales(name):
                return predictions, targets

            def transform(matrix):
                out = numpy.array(matrix, dtype="float64")
                for position in range(out.shape[1]):
                    out[:, position] = self.prep.rescale(name, out[:, position], set_name)
                return out
        return _rescale_tensor(predictions, transform), _rescale_tensor(targets, transform)

    def detached(self):
        """A copy for observation: the same batch, predictions detached from the graph."""
        copy = Context(self.batch, self.models, self.composites, self.emas, self.predicts, self.targets,
                       self.step, self.epoch, self.rng, self.train, self.prep, self.set_name, self.scaler,
                       self.losses, self.losses_keys, self.record)
        if self._outputs is not None:
            copy._outputs = {key: value.detach() if isinstance(value, torch.Tensor) else value
                             for key, value in self._outputs.items()}
        return copy


def _rescale_tensor(value, transform):
    if not value.is_floating_point():
        return value
    matrix = value.detach().float().cpu().numpy().reshape(len(value), -1)
    rescaled = numpy.asarray(transform(matrix))
    if rescaled.dtype.kind not in "fiu":
        return value
    return torch.from_numpy(rescaled.astype("float32").reshape(tuple(value.shape))).to(value.device)


def is_adapter(entry):
    return hasattr(entry, "tracker")


def entry_active(keys, set_name, turn):
    """Whether a definition runs on this set at this turn: its sets and every keys decide."""
    keys = keys or {}
    sets = keys.get("sets")
    if sets is not None and set_name not in sets:
        return False
    every = int(keys.get("every") or 1)
    return every <= 1 or turn % every == 0


def active_entries(table, keys_table, set_name, turn):
    """The (name, entry, keys) triples of a losses or metrics table active on a set at a turn."""
    keys_table = keys_table or {}
    return [(name, entry, keys_table.get(name) or {}) for name, entry in (table or {}).items()
            if entry_active(keys_table.get(name), set_name, turn)]


def resolve_entries(table, **extra):
    """Build the deferred params of every adapter in a table; objectives are left as they are."""
    for entry in (table or {}).values():
        resolver = getattr(entry, "resolve", None)
        if callable(resolver):
            resolver(**extra)


def parameter_names(fn):
    import inspect

    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


def objective_kwargs(fn, context):
    """The framework supplied parameters an objective's signature asks for (step, epoch, rng, scaler)."""
    names = parameter_names(fn)
    extra = {}
    if "step" in names:
        extra["step"] = context.step
    if "epoch" in names:
        extra["epoch"] = context.epoch
    if "rng" in names:
        extra["rng"] = context.rng
    if "scaler" in names:
        extra["scaler"] = context.scaler
    if "losses" in names:
        extra["losses"] = LossView(context)
    return extra


class LossView:
    """The other losses of the table evaluated on the current batch by name, for objectives that combine them."""

    def __init__(self, context):
        self.context = context

    def __contains__(self, name):
        return name in (self.context.losses or {})

    def __getitem__(self, name):
        table = self.context.losses or {}
        if name not in table:
            raise KeyError(f"losses has no definition {name!r}; the definitions are {sorted(table)}")
        keys = (self.context.losses_keys or {}).get(name)
        return entry_loss(table[name], self.context, keys)


def entry_loss(entry, context, keys=None):
    """The loss tensor of a losses entry: an adapter reads the context, an objective takes models and batch."""
    if is_adapter(entry):
        return entry.loss(context, keys)
    return entry(context.everything(), context.batch, **objective_kwargs(entry, context))


def turn_generator(device=None):
    """A torch generator on the device, seeded from the global RNG so seeded runs stay deterministic."""
    target = torch.device(device or "cpu")
    generator = torch.Generator(device=target)
    generator.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,))))
    return generator


def amp_context(device, enabled):
    """Autocast for mixed precision: float16 on cuda, bfloat16 on the cpu; a no op when disabled."""
    import contextlib

    if not enabled:
        return contextlib.nullcontext()
    kind = torch.device(device or "cpu").type
    return torch.autocast(device_type=kind, dtype=torch.float16 if kind == "cuda" else torch.bfloat16)


def amp_scaler(device, enabled):
    """The loss scaler of mixed precision; enabled on cuda only (bfloat16 on the cpu needs none)."""
    if not enabled:
        return None
    kind = torch.device(device or "cpu").type
    return torch.amp.GradScaler(kind, enabled=kind == "cuda")


def loss_scalar(value):
    if isinstance(value, dict):
        if "loss" not in value:
            raise ValueError(f"an objective returning a mapping must include a 'loss' term, got {sorted(value)}")
        return value["loss"]
    return value


class ObjectiveTracker:
    def __init__(self, entry, name):
        self.entry = entry
        self.name = name
        self.totals = {}
        self.count = 0

    def observe(self, context):
        value = self.entry(context.everything(), context.batch, **objective_kwargs(self.entry, context))
        self.record(value, context.size)

    def record(self, value, size):
        terms = value if isinstance(value, dict) else {"loss": value}
        for term, item in terms.items():
            number = float(item.detach()) if isinstance(item, torch.Tensor) else float(item)
            self.totals[term] = self.totals.get(term, 0.0) + number * size
        self.count += size

    def result(self):
        if not self.count:
            return {self.name: math.nan}
        out = {}
        for term, total in self.totals.items():
            key = self.name if term == "loss" else f"{self.name}/{term}"
            out[key] = total / self.count
        return out


def tracker_for(name, entry, keys=None, rescale=False):
    """A tracker for a table entry; ``rescale`` (the metrics table) reports in the original scale."""
    if is_adapter(entry):
        return entry.tracker(name, keys, rescale)
    return ObjectiveTracker(entry, name)


def observe_all(trackers, context):
    with torch.no_grad():
        for tracker in trackers:
            tracker.observe(context)


def collect_results(trackers):
    merged = {}
    for tracker in trackers:
        merged.update(tracker.result())
    return merged


def module_trainable(module):
    return getattr(module, "kalfa_trainable", True)


def set_modes(models, train, composites=None):
    """Trained models to train or eval mode (untrainable ones stay in eval); composites follow the phase."""
    for model in models.values():
        if train and module_trainable(model):
            model.train()
        else:
            model.eval()
    for composite in (composites or {}).values():
        composite.train(bool(train))
