import fnmatch
import hashlib

import torch
from torch import nn

from kalfa.registration import lego
from kalfa.std.builder.base import Model, weights_path
from kalfa.std.checkpoint.base import load
from kalfa.std.common.deferred import DeferredLayer
from kalfa.std.common.log import clock, logger_for, since
from kalfa.std.layer.base import LazyLayer


logger = logger_for("models")


def is_lazy(module):
    return isinstance(module, (nn.modules.lazy.LazyModuleMixin, LazyLayer))


def model_seed(seed, index):
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63)


def role_of(name, parameter):
    if parameter.dim() >= 2:
        return "weights"
    if "bias" in name:
        return "bias"
    return "scale"


def apply_roles(root, roles, patterns=()):
    owners = {}
    for module_name, module in root.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            owners[f"{module_name}.{name}" if module_name else name] = (name, parameter)
    for full, (name, parameter) in owners.items():
        initializer = (roles or {}).get(role_of(name, parameter))
        if initializer is not None:
            with torch.no_grad():
                initializer(parameter)
    for entry in patterns or []:
        for full, (name, parameter) in owners.items():
            if not fnmatch.fnmatchcase(full, entry.get("match")):
                continue
            initializer = entry.get(role_of(name, parameter))
            if initializer is not None:
                with torch.no_grad():
                    initializer(parameter)


class Seeded:
    def __init__(self, context, seed):
        self.context = context
        self.seed = seed

    def __enter__(self):
        self.context.__enter__()
        torch.manual_seed(self.seed)
        return self

    def __exit__(self, *error):
        return self.context.__exit__(*error)


class Module(Model):
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
        self.trainable = bool(trainable)
        self.weights = weights
        self.pending_state = None
        self.initialized = False
        self.settled = False
        if not any(is_lazy(module) for module in self.nodes.modules()):
            self.build()

    def seeded(self):
        if self.seed is None:
            return torch.random.fork_rng(devices=[], enabled=False)
        return Seeded(torch.random.fork_rng(devices=[]), self.seed)

    def reset_parameters(self):
        for module in self.nodes.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset) and not is_lazy(module):
                reset()

    def build(self):
        with self.seeded():
            if self.seed is not None:
                self.reset_parameters()
            roles = {role: function for role, function in self.init.items() if role != "patterns"}
            apply_roles(self.nodes, roles, self.init.get("patterns"))
            for name, spec in self.node_init.items():
                roles = {role: function for role, function in (spec or {}).items() if role != "patterns"}
                apply_roles(self.nodes[self.safe[name]], roles, (spec or {}).get("patterns"))
        self.initialized = True
        if self.pending_state is not None:
            state, self.pending_state = self.pending_state, None
            self.load(state)
        self.settle()

    def load(self, state):
        try:
            super().load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError(f"the weights do not fit the model: {error}") from None

    def settle(self):
        for parameter in self.nodes.parameters():
            parameter.requires_grad_(self.trainable)
        if not self.trainable:
            self.eval()
        self.settled = True

    def train(self, mode=True):
        if mode and not self.trainable:
            mode = False
        return super().train(mode)

    def materialize(self, arguments):
        was_training = self.training
        super().train(False)
        with self.seeded(), torch.no_grad():
            self.run(arguments)
        super().train(was_training)
        self.build()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        modules = list(self.nodes.modules())
        if not self.initialized and any(isinstance(module, LazyLayer) for module in modules):
            self.pending_state = dict(state_dict)
            return None
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.initialized = True
        self.settle()
        return result

    def forward(self, *arguments):
        if not self.initialized:
            self.materialize(arguments)
        return self.run(arguments)

    def run(self, arguments):
        if len(arguments) != len(self.inputs):
            raise ValueError(f"model takes {len(self.inputs)} inputs {self.inputs}, got {len(arguments)}")
        values = dict(zip(self.inputs, arguments))
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


def load_weights(spec):
    path = weights_path(spec)
    if not path.exists():
        raise FileNotFoundError(f"weights: {path} does not exist")
    states = load(path).get("models", {})
    if spec.get("model") not in states:
        raise KeyError(f"weights: run {spec['run']!r} has no model {spec.get('model')!r}; it has {sorted(states)}")
    return states[spec["model"]]


@lego("/builder/kalfa/module", bus=["prep", "train_loader"], roles=["weights", "bias", "scale"],
      description="Build a model graph into an nn.Module under hash(seed, index), apply init roles, "
                  "trainable and weights; reference nodes take the models dict; layer params that are kind "
                  "data components are built from prep and the train loader")
def module(graph, seed=None, index=0, init=None, trainable=True, weights=None, models=None, prep=None,
           train_loader=None):
    started = clock()
    built = Module(graph, seed, index, init, trainable, weights, models, prep, train_loader)
    if weights is not None:
        state = load_weights(weights)
        logger.info(f"weights from {weights_path(weights)}")
        if built.initialized:
            built.load(state)
            built.settle()
        else:
            built.pending_state = dict(state)
    logger.debug(f"built a module under seed {seed} index {index} ({since(started)})")
    return built
