"""Optimizers: legos that build a torch optimizer over the union of the models that chose it."""

import fnmatch

import torch

from ..registration import lego


class LazyOptimizer:
    """A torch optimizer created on first use, once every lazy layer has its parameters.

    Carries the name of the loss it minimizes and its schedule; zero_grad clears only its own parameters.
    """

    def __init__(self, factory, models, params, schedule, loss):
        self.factory = factory
        self.models = dict(models)
        self.params = dict(params or {})
        self.schedule = schedule
        self.loss = loss
        self.real = None
        self.pending = None
        self.updates = 0
        self.base = None

    def named_parameters(self):
        for model_name, model in self.models.items():
            for name, parameter in model.named_parameters():
                yield f"{model_name}.{name}", parameter

    def parameters(self):
        return [parameter for _, parameter in self.named_parameters()]

    def _groups(self):
        groups = self.params.get("groups") or []
        defaults = {key: value for key, value in self.params.items() if key != "groups"}
        buckets = [[] for _ in groups]
        rest = []
        for name, parameter in self.named_parameters():
            for position, group in enumerate(groups):
                if fnmatch.fnmatchcase(name, group.get("match", "")):
                    buckets[position].append(parameter)
                    break
            else:
                rest.append(parameter)
        entries = [{"params": rest}]
        for group, bucket in zip(groups, buckets):
            entries.append({"params": bucket, **{key: value for key, value in group.items() if key != "match"}})
        return entries, defaults

    def _ensure(self):
        if self.real is None:
            entries, defaults = self._groups()
            if not entries[0]["params"] and len(entries) == 1:
                raise ValueError("the optimizer has no parameters; its models have none yet")
            self.real = self.factory([entry for entry in entries if entry["params"]], defaults)
            self.base = [group["lr"] for group in self.real.param_groups]
            if self.pending is not None:
                self.real.load_state_dict(self.pending)
                self.pending = None
            self._schedule()
        return self.real

    def torch(self):
        """The torch optimizer itself, for loss scalers and schedulers."""
        return self._ensure()

    def _schedule(self):
        if self.schedule is None or self.real is None:
            return
        factor = float(self.schedule(self.updates))
        for group, base in zip(self.real.param_groups, self.base):
            group["lr"] = base * factor

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = None

    def step(self):
        self._ensure().step()
        self.updates += 1
        self._schedule()

    @property
    def param_groups(self):
        return self._ensure().param_groups

    def lr(self):
        if self.real is None:
            return float(self.params.get("lr", math_nan()))
        return float(self.real.param_groups[0]["lr"])

    def set_param(self, name, value):
        self.params[name] = value
        if self.real is not None:
            for position, group in enumerate(self.real.param_groups):
                group[name] = value
                if name == "lr":
                    self.base[position] = value
            self._schedule()

    def state_dict(self):
        inner = self.pending if self.real is None else self.real.state_dict()
        return {"torch": inner, "updates": self.updates, "base": self.base}

    def load_state_dict(self, state):
        if state is None:
            return
        if isinstance(state, dict) and "torch" in state and "updates" in state:
            self.updates = int(state["updates"])
            if state.get("base") is not None:
                self.base = list(state["base"])
            state = state["torch"]
        if state is None:
            return
        if self.real is None:
            self.pending = state
        else:
            self.real.load_state_dict(state)
            self._schedule()


def math_nan():
    return float("nan")


def _make(torch_class):
    def factory(entries, defaults):
        return torch_class(entries, **defaults)
    return factory


@lego("/optimizer/torch/adam", state=True, refs={"loss": "loss", "schedule": "schedule"},
            aliases="models", alias="adam", description="torch Adam over the union of its models; params are Adam's keyword arguments")
def adam(models, params=None, schedule=None, loss=None):
    return LazyOptimizer(_make(torch.optim.Adam), models, params, schedule, loss)


@lego("/optimizer/torch/adamw", state=True, refs={"loss": "loss", "schedule": "schedule"},
            aliases="models", alias="adamw", description="torch AdamW over the union of its models")
def adamw(models, params=None, schedule=None, loss=None):
    return LazyOptimizer(_make(torch.optim.AdamW), models, params, schedule, loss)


@lego("/optimizer/torch/sgd", state=True, refs={"loss": "loss", "schedule": "schedule"},
            aliases="models", alias="sgd", description="torch SGD over the union of its models")
def sgd(models, params=None, schedule=None, loss=None):
    return LazyOptimizer(_make(torch.optim.SGD), models, params, schedule, loss)
