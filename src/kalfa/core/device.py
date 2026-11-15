import contextlib

import torch

from .errors import RunError


class Device:
    def __init__(self, spec="auto"):
        self.spec = spec
        self.torch = self.pick(spec)

    @staticmethod
    def pick(spec):
        if isinstance(spec, torch.device):
            return spec
        if isinstance(spec, Device):
            return spec.torch
        if isinstance(spec, int) and not isinstance(spec, bool):
            return torch.device("cuda", spec)
        text = str(spec or "auto").strip().lower()
        if text == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        try:
            return torch.device(text)
        except RuntimeError as error:
            raise RunError(f"device {spec!r} is not understood: {error}") from None

    @property
    def type(self):
        return self.torch.type

    def move(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.torch, non_blocking=self.type == "cuda")
        if isinstance(batch, dict):
            return {key: self.move(value) for key, value in batch.items()}
        if isinstance(batch, list):
            return [self.move(value) for value in batch]
        if isinstance(batch, tuple):
            return tuple(self.move(value) for value in batch)
        return batch

    def autocast(self, enabled):
        if not enabled or self.type == "cpu":
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.type)

    def scaler(self, enabled):
        if enabled and self.type == "cuda":
            return torch.amp.GradScaler("cuda")
        return None

    def note(self):
        name = torch.cuda.get_device_name(self.torch) if self.type == "cuda" and torch.cuda.is_available() else None
        return {"type": self.type, "name": name, "spec": str(self.spec)}

    def __str__(self):
        return str(self.torch)
