import hashlib
import random

import numpy
import torch


class Rng:
    def __init__(self, seed=0):
        self.seed = int(seed)
        self.streams = {}
        self.pending = {}

    def seed_everything(self):
        random.seed(self.seed)
        numpy.random.seed(self.seed % (2**32))
        torch.manual_seed(self.seed)

    def seed_for(self, name):
        digest = hashlib.sha256(f"{self.seed}:{name}".encode()).digest()
        return int.from_bytes(digest[:8], "little") % (2**63 - 1)

    @staticmethod
    def place(device=None, like=None):
        if device is None and like is not None:
            device = like.device
        if device is None:
            return torch.device("cpu")
        if isinstance(device, torch.device):
            return device
        found = getattr(device, "torch", None)
        if isinstance(found, torch.device):
            return found
        return torch.device(str(device))

    @staticmethod
    def size(shape, like=None):
        if shape is None:
            return tuple(like.shape)
        if isinstance(shape, int):
            return (shape,)
        return tuple(shape)

    @staticmethod
    def float_dtype(dtype, like):
        if dtype is not None:
            return dtype
        if like is not None and like.dtype.is_floating_point:
            return like.dtype
        return None

    def stream(self, name, device=None):
        where = self.place(device)
        key = f"{name}@{where.type}"
        generator = self.streams.get(key)
        if generator is None:
            generator = torch.Generator(device=where)
            generator.manual_seed(self.seed_for(name))
            state = self.pending.pop(key, None)
            if state is not None:
                generator.set_state(state)
            self.streams[key] = generator
        return generator

    def normal(self, name, shape, like=None, device=None, dtype=None):
        where = self.place(device, like)
        return torch.randn(self.size(shape, like), generator=self.stream(name, where), device=where,
                           dtype=self.float_dtype(dtype, like))

    def uniform(self, name, shape, like=None, device=None, dtype=None):
        where = self.place(device, like)
        return torch.rand(self.size(shape, like), generator=self.stream(name, where), device=where,
                          dtype=self.float_dtype(dtype, like))

    def integers(self, name, low, high, shape, device=None):
        where = self.place(device)
        return torch.randint(int(low), int(high), self.size(shape), generator=self.stream(name, where), device=where)

    def permutation(self, name, count, device=None):
        where = self.place(device)
        return torch.randperm(int(count), generator=self.stream(name, where), device=where)

    def state_dict(self):
        streams = {key: generator.get_state().detach().cpu().clone() for key, generator in self.streams.items()}
        for key, state in self.pending.items():
            streams.setdefault(key, state)
        return {"seed": self.seed, "streams": streams}

    def load_state_dict(self, state):
        if not state:
            return
        if state.get("seed") is not None:
            self.seed = int(state["seed"])
        for key, saved in (state.get("streams") or {}).items():
            tensor = (saved if isinstance(saved, torch.Tensor) else torch.as_tensor(saved)).cpu()
            generator = self.streams.get(key)
            if generator is not None:
                generator.set_state(tensor)
            else:
                self.pending[key] = tensor
