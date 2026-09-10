import copy
import random
from pathlib import Path

import numpy
import torch


class Policy:
    monitor: str | None = None

    def tags(self, metrics: dict) -> list:
        raise NotImplementedError

    def state(self) -> dict | None:
        return None

    def restore(self, state: dict | None) -> None:
        pass


def rng_states():
    states = {"python": random.getstate(), "numpy": numpy.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng(states):
    if not states:
        return
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        numpy.random.set_state(states["numpy"])
    if "torch" in states:
        torch.set_rng_state(states["torch"])
    if states.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def state_dicts(items):
    return {name: item.state_dict() for name, item in (items or {}).items()}


def payload(models, optimizers, emas, counters, rules, checkpoint=None):
    return {"models": state_dicts(models), "optimizers": state_dicts(optimizers), "emas": state_dicts(emas),
            "counters": dict(counters or {}), "rules": copy.deepcopy(dict(rules or {})),
            "checkpoint": checkpoint, "rng": rng_states(), "turn": int((counters or {}).get("turn", 0))}


def save(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, target)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_into(models, optimizers, emas, counters, rules, data):
    for name, state in data.get("models", {}).items():
        if name in models:
            models[name].load_state_dict(state)
    for name, state in data.get("emas", {}).items():
        if name in emas:
            emas[name].load_state_dict(state)
    for name, state in data.get("optimizers", {}).items():
        if name in optimizers:
            optimizers[name].load_state_dict(state)
    counters.clear()
    counters.update(data.get("counters", {}))
    rules.clear()
    rules.update(copy.deepcopy(data.get("rules", {})))
    if data.get("checkpoint") is not None:
        rules["checkpoint"] = data["checkpoint"]
    restore_rng(data.get("rng"))
