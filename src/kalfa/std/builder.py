"""The model builder: a cirak graph becomes an nn.Module with kalfa's seed, init, trainable and weights rules."""

import fnmatch
import hashlib

import torch
from torch import nn

from ..registration import lego
from .deferred import DeferredLayer
from .log import clock, logger, since

LOG = logger("models")

NORMALIZATION = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm)


def is_lazy(module):
    return isinstance(module, nn.modules.lazy.LazyModuleMixin) or getattr(module, "kalfa_lazy", False)


def model_seed(seed, index):
    """A distinct seed per model index derived from the global seed; adding a model changes no other."""
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63)


def role_of(module, name, parameter):
    if parameter.dim() >= 2:
        return "weights"
    if name.endswith("bias"):
        return "bias"
    if isinstance(module, NORMALIZATION) and name.endswith("weight"):
        return "scale"
    return None


def apply_roles(root, roles, patterns=()):
    """Apply role initializers, then the pattern list in order, to the parameters of ``root``."""
    owners = {}
    for module_name, module in root.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            owners[f"{module_name}.{name}" if module_name else name] = (module, name, parameter)
    for full, (module, name, parameter) in owners.items():
        role = role_of(module, name, parameter)
        initializer = (roles or {}).get(role)
        if initializer is not None:
            with torch.no_grad():
                initializer(parameter)
    for entry in patterns or []:
        match = entry.get("match")
        for full, (module, name, parameter) in owners.items():
            if not fnmatch.fnmatchcase(full, match):
                continue
            role = role_of(module, name, parameter)
            initializer = entry.get(role)
            if initializer is not None:
                with torch.no_grad():
                    initializer(parameter)


class Module(nn.Module):
    """A model graph as a module: nodes run in order over named wires; reference nodes call other models."""

    def __init__(self, graph, seed=None, index=0, init=None, trainable=True, weights=None, models=None, prep=None,
                 train_loader=None):
        super().__init__()
        self.graph = graph
        self.inputs = list(graph.inputs)
        self.outputs = list(graph.outputs)
        self.safe = {node.name: node.name.replace(".", "__") for node in graph.nodes}
        available = {"prep": prep, "loader": train_loader, "train_loader": train_loader}
        objects = {node.name: node.obj.build(**available) if isinstance(node.obj, DeferredLayer) else node.obj
                   for node in graph.nodes if node.ref is None}
        self.nodes = nn.ModuleDict({self.safe[name]: obj for name, obj in objects.items()})
        object.__setattr__(self, "refs", {})
        for node in graph.nodes:
            if node.ref is not None:
                if models is None or node.ref not in models:
                    raise KeyError(f"node {node.name!r} references model {node.ref!r}, which is not built")
                self.refs[node.name] = models[node.ref]
        self.seed = None if seed is None else model_seed(seed, index)
        self.init = dict(init or {})
        self.node_init = {node.name: node.extra["init"] for node in graph.nodes if "init" in node.extra}
        self.kalfa_trainable = bool(trainable)
        self.weights = weights
        self.pending_state = None
        self.initialized = False
        self.settled = False
        if not any(is_lazy(module) for module in self.nodes.modules()):
            self._build()

    def _seeded(self):
        if self.seed is None:
            return torch.random.fork_rng(devices=[], enabled=False)
        context = torch.random.fork_rng(devices=[])
        return _Seeded(context, self.seed)

    def _build(self):
        with self._seeded():
            if self.seed is not None:
                for module in self.nodes.modules():
                    reset = getattr(module, "reset_parameters", None)
                    if callable(reset) and not is_lazy(module):
                        reset()
            roles = {role: fn for role, fn in self.init.items() if role != "patterns"}
            apply_roles(self.nodes, roles, self.init.get("patterns"))
            for name, spec in self.node_init.items():
                roles = {role: fn for role, fn in (spec or {}).items() if role != "patterns"}
                apply_roles(self.nodes[self.safe[name]], roles, (spec or {}).get("patterns"))
        self.initialized = True
        if self.pending_state is not None:
            state, self.pending_state = self.pending_state, None
            self._load(state)
        self._settle()

    def _load(self, state):
        try:
            super().load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"the weights do not fit the model: {exc}") from None

    def _settle(self):
        for parameter in self.nodes.parameters():
            parameter.requires_grad_(self.kalfa_trainable)
        if not self.kalfa_trainable:
            self.eval()
        self.settled = True

    def train(self, mode=True):
        if mode and not self.kalfa_trainable:
            mode = False
        return super().train(mode)

    def _materialize(self, args):
        was_training = self.training
        super().train(False)
        with self._seeded(), torch.no_grad():
            self._run(args)
        super().train(was_training)
        self._build()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not self.initialized and any(is_lazy(module) for module in self.nodes.modules()) \
                and not all(isinstance(module, nn.modules.lazy.LazyModuleMixin) or not is_lazy(module)
                            for module in self.nodes.modules()):
            self.pending_state = dict(state_dict)
            return None
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.initialized = True
        self._settle()
        return result

    def forward(self, *args):
        if not self.initialized:
            self._materialize(args)
        return self._run(args)

    def _run(self, args):
        if len(args) != len(self.inputs):
            raise ValueError(f"model takes {len(self.inputs)} inputs {self.inputs}, got {len(args)}")
        values = dict(zip(self.inputs, args))
        for node in self.graph.nodes:
            inputs = [values[wire] for wire in node.inputs]
            if node.ref is not None:
                result = self.refs[node.name](*inputs)
            else:
                result = self.nodes[self.safe[node.name]](*inputs)
            if node.unpack:
                if not isinstance(result, (tuple, list)) or len(result) < len(node.outputs):
                    raise ValueError(f"node {node.name!r} declares outputs {list(node.outputs)} but returned "
                                     f"{type(result).__name__}")
                values.update(zip(node.outputs, result))
            else:
                values[node.outputs[0]] = result
        if len(self.outputs) == 1:
            return values[self.outputs[0]]
        return tuple(values[wire] for wire in self.outputs)


class _Seeded:
    def __init__(self, context, seed):
        self.context = context
        self.seed = seed

    def __enter__(self):
        self.context.__enter__()
        torch.manual_seed(self.seed)
        return self

    def __exit__(self, *exc):
        return self.context.__exit__(*exc)


WEIGHT_FILES = {"best": ("checkpoints", "best.pt"), "last": ("checkpoints", "last.pt"),
                "final": ("final", "state.pt")}


def weights_path(spec):
    """The checkpoint file a weights spec {run, model, which} names."""
    from pathlib import Path

    which = spec.get("which")
    if which not in WEIGHT_FILES:
        raise ValueError(f"weights.which must be best, last or final, got {which!r}")
    return Path(spec["run"]).joinpath(*WEIGHT_FILES[which])


def load_weights(spec):
    """The state dict of the named model in the named run's checkpoint."""
    from .checkpoint import load

    path = weights_path(spec)
    if not path.exists():
        raise FileNotFoundError(f"weights: {path} does not exist")
    payload = load(path)
    states = payload.get("models", {})
    if spec.get("model") not in states:
        raise KeyError(f"weights: run {spec['run']!r} has no model {spec.get('model')!r}; it has {sorted(states)}")
    return states[spec["model"]]


@lego("/builder/kalfa/module", bus=["prep", "train_loader"],
            description="Build a model graph into an nn.Module under hash(seed, index), apply init roles, "
                        "trainable and weights; reference nodes take the models dict; layer params that are kind "
                        "data components are built from prep and the train loader")
def module(graph, seed=None, index=0, init=None, trainable=True, weights=None, models=None, prep=None,
           train_loader=None):
    started = clock()
    built = Module(graph, seed, index, init, trainable, weights, models, prep, train_loader)
    if weights is not None:
        state = load_weights(weights)
        LOG.info(f"weights from {weights_path(weights)}")
        if built.initialized:
            built._load(state)
            built._settle()
        else:
            built.pending_state = dict(state)
    LOG.debug(f"built a module under seed {seed} index {index} ({since(started)})")
    return built
