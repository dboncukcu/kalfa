import contextlib

import torch


class Device:
    def __init__(self, target, uri, params=None):
        self.torch = torch.device(target)
        self.uri = uri
        self.params = dict(params or {})

    @classmethod
    def cpu(cls):
        return cls("cpu", "/device/kalfa/cpu")

    @property
    def type(self):
        return self.torch.type

    def move(self, batch):
        return {key: value.to(self.torch) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()}

    def place(self, modules):
        for module in modules.values():
            module.to(self.torch)
        return modules

    def generator(self):
        generator = torch.Generator(device=self.torch)
        generator.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,))))
        return generator

    def autocast(self, enabled):
        if not enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.type, dtype=torch.float16 if self.type == "cuda" else torch.bfloat16)

    def scaler(self, enabled):
        if not enabled:
            return None
        return torch.amp.GradScaler(self.type, enabled=self.type == "cuda")

    def note(self):
        return {"device": str(self.torch), "uri": self.uri, "params": self.params}

    def __str__(self):
        return str(self.torch)

    def __eq__(self, other):
        return isinstance(other, Device) and self.torch == other.torch and self.uri == other.uri \
            and self.params == other.params
