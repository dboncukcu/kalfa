import fnmatch
import math

from kalfa.std.common.effects import relative, relative_effect


class Optimizer:
    torch_class = None

    def __init__(self, models, params=None, schedule=None, loss=None):
        self.models = dict(models)
        self.params = dict(params or {})
        if "groups" in self.params:
            self.params["groups"] = [dict(group) for group in (self.params["groups"] or [])]
        self.schedule = schedule
        self.loss = loss
        self.real = None
        self.pending = None
        self.updates = 0
        self.base = None
        self.placed = None

    @property
    def name(self) -> str:
        return self.torch_class.__name__.lower()

    def named_parameters(self):
        for model_name, model in self.models.items():
            for name, parameter in model.named_parameters():
                yield f"{model_name}.{name}", parameter

    def parameters(self):
        return [parameter for _, parameter in self.named_parameters()]

    def group_specs(self):
        return self.params.get("groups") or []

    def groups(self):
        groups = self.group_specs()
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
            entries.append({"params": bucket,
                            **{key: value for key, value in group.items() if key not in ("match", "name")}})
        return entries, defaults

    def torch(self):
        if self.real is not None:
            return self.real
        entries, defaults = self.groups()
        kept = [position for position, entry in enumerate(entries) if entry["params"]]
        if not kept:
            raise ValueError("the optimizer has no parameters; its models have none yet")
        self.placed = [kept.index(position) if position in kept else None for position in range(len(entries))]
        self.real = self.torch_class([entries[position] for position in kept], **defaults)
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

    def rates(self):
        found = {}
        for position, group in enumerate(self.group_specs()):
            name = group.get("name")
            if name is None:
                continue
            if self.real is None:
                value = self.value_of(position + 1, "lr")
                found[name] = math.nan if value is None else float(value)
            elif self.placed[position + 1] is not None:
                found[name] = float(self.real.param_groups[self.placed[position + 1]]["lr"])
        return found

    def slots_of(self, target):
        groups = self.group_specs()
        if target is None:
            return [0]
        if target == "*":
            return list(range(len(groups) + 1))
        found = [position + 1 for position, group in enumerate(groups)
                 if group.get("name") is not None and fnmatch.fnmatchcase(str(group["name"]), target)]
        if not found:
            names = [group["name"] for group in groups if group.get("name") is not None]
            raise KeyError(f"the optimizer has no group named {target!r}; the named groups are {names}")
        return found

    def value_of(self, slot, name):
        if slot == 0:
            return self.params.get(name)
        group = self.group_specs()[slot - 1]
        return group[name] if name in group else self.params.get(name)

    def write(self, slot, name, value):
        if slot == 0:
            self.params[name] = value
            slots = [0, *(position + 1 for position, group in enumerate(self.group_specs()) if name not in group)]
        else:
            self.group_specs()[slot - 1][name] = value
            slots = [slot]
        if self.real is None:
            return
        for each in slots:
            index = self.placed[each]
            if index is None:
                continue
            self.real.param_groups[index][name] = value
            if name == "lr":
                self.base[index] = value

    def set_param(self, name, value, target=None):
        selected = self.slots_of(target)
        for slot in selected:
            if slot != 0 and 0 in selected and name not in self.group_specs()[slot - 1]:
                continue
            current = self.value_of(slot, name)
            if isinstance(value, dict):
                if not relative_effect(value):
                    raise KeyError(f"a relative effect is {{times: x}} or {{plus: x}} with a number, got "
                                   f"{sorted(value)}")
                if current is None:
                    raise KeyError(f"a relative effect on {name!r} needs a value to change, and the optimizer has "
                                   f"no {name!r}")
                self.write(slot, name, relative(float(current), value))
            else:
                self.write(slot, name, value)
        if self.real is not None:
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
