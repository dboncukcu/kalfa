from torch import nn


class Objective(nn.Module):
    on = ("train", "valid", "test")
    every = 1
    fixed = ()
    name = ""
    path = ""

    def bind(self, ctx):
        pass

    def forward(self, outputs, batch, ctx):
        raise NotImplementedError


class ContextMetric:
    on = ("train", "valid", "test")
    every = 1
    name = ""
    path = ""

    def bind(self, ctx):
        pass

    def reset(self):
        raise NotImplementedError

    def update(self, outputs, batch, ctx):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class Step:
    order = None
    amp = False
    grad_clip = None
    accumulate = 1
    stepped = True
    fixed = ()
    persist = ()
    name = "step"
    path = "train.step"

    def bind(self, ctx):
        pass

    def train_turn(self, ctx):
        raise NotImplementedError

    def train_batch(self, batch, ctx):
        raise NotImplementedError

    def eval_batch(self, batch, ctx):
        raise NotImplementedError

    def predict(self, batch, ctx):
        raise NotImplementedError

    def state_dict(self):
        return {name: getattr(self, name) for name in self.persist}

    def load_state_dict(self, state):
        for name, value in (state or {}).items():
            setattr(self, name, value)


class Trigger:
    persist = ()
    fixed = ()
    name = ""
    path = ""

    def bind(self, ctx):
        pass

    def fires(self, ctx):
        raise NotImplementedError

    def reset(self):
        pass

    def describe(self):
        return type(self).__name__

    def state_dict(self):
        return {name: getattr(self, name) for name in self.persist}

    def load_state_dict(self, state):
        for name, value in (state or {}).items():
            setattr(self, name, value)


class Action:
    name = ""
    path = ""

    def bind(self, ctx):
        pass

    def __call__(self, ctx):
        raise NotImplementedError


class Logger:
    name = ""
    path = ""

    def bind(self, ctx):
        pass

    def turn(self, ctx, line):
        pass

    def update(self, ctx, line):
        pass

    def close(self, ctx):
        pass
