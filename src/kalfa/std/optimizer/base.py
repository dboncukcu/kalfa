import fnmatch
import math


class Optimizer:
    torch_class = None

    def __init__(self, models, params=None, schedule=None, loss=None):
        self.models = dict(models)
        self.params = dict(params or {})
        self.schedule = schedule
        self.loss = loss
        self.real = None
        self.pending = None
        self.updates = 0
        self.base = None

    @property
    def name(self) -> str:
        return self.torch_class.__name__.lower()

    def named_parameters(self):
        for model_name, model in self.models.items():
            for name, parameter in model.named_parameters():
                yield f"{model_name}.{name}", parameter

    def parameters(self):
        return [parameter for _, parameter in self.named_parameters()]

    def groups(self):
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

    def torch(self):
        if self.real is not None:
            return self.real
        entries, defaults = self.groups()
        if not entries[0]["params"] and len(entries) == 1:
            raise ValueError("the optimizer has no parameters; its models have none yet")
        self.real = self.torch_class([entry for entry in entries if entry["params"]], **defaults)
        self.base = [group["lr"] for group in self.real.param_groups]
        if self.pending is not None:
            self.real.load_state_dict(self.pending)
            self.pending = None
        self.reschedule()
        return self.real

    def reschedule(self):
        if self.schedule is None or self.real is None:
            return
        factor = float(self.schedule(self.updates))
        for group, base in zip(self.real.param_groups, self.base):
            group["lr"] = base * factor

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = None

    def step(self):
        self.torch().step()
        self.updates += 1
        self.reschedule()

    @property
    def param_groups(self):
        return self.torch().param_groups

    def lr(self) -> float:
        if self.real is None:
            return float(self.params.get("lr", math.nan))
        return float(self.real.param_groups[0]["lr"])

    def set_param(self, name, value):
        self.params[name] = value
        if self.real is not None:
            for position, group in enumerate(self.real.param_groups):
                group[name] = value
                if name == "lr":
                    self.base[position] = value
            self.reschedule()

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
            self.reschedule()
