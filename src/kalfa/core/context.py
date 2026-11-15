import logging
import time
from dataclasses import dataclass, field

from .errors import KalfaError, RunError


def as_list(value):
    if value is None or value is False:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def train_of(tree):
    settings = getattr(tree, "settings", None)
    train = settings.get("train") if isinstance(settings, dict) else None
    return train if isinstance(train, dict) else {}


@dataclass
class TrainSettings:
    epochs: int | None = None
    updates: dict | None = None
    predicts: str | None = None
    evaluate: list = field(default_factory=list)
    checkpoint: dict | None = None
    report: str = "last"
    track: list = field(default_factory=list)
    log: list = field(default_factory=list)

    @classmethod
    def from_mapping(cls, mapping):
        mapping = dict(mapping or {})
        epochs = mapping.get("epochs")
        updates = mapping.get("updates")
        checkpoint = mapping.get("checkpoint")
        return cls(
            epochs=None if epochs is None else int(epochs),
            updates=None if updates is None else dict(updates),
            predicts=mapping.get("predicts"),
            evaluate=as_list(mapping.get("evaluate", ["valid", "test"])),
            checkpoint=dict(checkpoint) if isinstance(checkpoint, dict) else checkpoint,
            report=str(mapping.get("report") or "last"),
            track=as_list(mapping.get("track")),
            log=as_list(mapping.get("log")),
        )

    @property
    def unit(self):
        return "updates" if self.updates else "epochs"

    def total_turns(self):
        if self.updates:
            total = int(self.updates["total"])
            per_turn = int(self.updates.get("turn", total))
            return max(1, -(-total // per_turn))
        return int(self.epochs or 0)


class Context:
    def __init__(self, tree, plan=None):
        self.tree = tree
        self.config = tree.config
        self.plan = plan
        self.settings = train_of(tree)
        self.train = TrainSettings.from_mapping(self.settings)
        self.data = None
        self.models = {}
        self.losses = {}
        self.metrics = {}
        self.optimizers = {}
        self.step = None
        self.rules = []
        self.plots = {}
        self.generate = None
        self.loggers = []
        self.device = None
        self.rng = None
        self.seed = 0
        self.record = None
        self.history = []
        self.turn = 0
        self.updates = 0
        self.batches = 0
        self.set_name = None
        self.latest = {}
        self.fired = []
        self.stopping = None
        self.stopped_by = None
        self.stop_noted = False
        self.first_calls = None
        self.monitor = None
        self.best = None
        self.cursor = None
        self.last_norms = {}
        self.patches = []
        self.fitted_from = None
        self.want_sets = True
        self.rebuild_failure = None
        self.started = time.perf_counter()
        self.logger = logging.getLogger("kalfa.run")

    def predicts(self):
        name = self.train.predicts if self.train is not None else (self.settings or {}).get("predicts")
        if name is False:
            return None
        if name:
            name = str(name)
            for prefix in ("@models.", "models."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            entry = self.models.get(name)
            if entry is None:
                raise RunError(f"train.predicts names the model {name!r}; the models are {sorted(self.models)}")
            return entry
        if len(self.models) == 1:
            return next(iter(self.models.values()))
        return None

    def stop(self, reason=None, by=None):
        self.stopping = reason or "stop"
        if by is not None:
            self.stopped_by = by

    def elapsed(self):
        return time.perf_counter() - self.started

    def loaders(self):
        if self.data is None:
            return {}
        try:
            return self.data.loaders()
        except KalfaError:
            if self.rebuild_failure is None:
                raise
            raise RunError(f"the sets of {self.fitted_from} could not be rebuilt: {self.rebuild_failure}") from None

    def set_modes(self, train):
        train = bool(train)
        for objective in self.losses.values():
            if hasattr(objective, "train"):
                objective.train(train)
        for entry in self.models.values():
            if callable(getattr(entry, "set_mode", None)):
                entry.set_mode(train)
            else:
                entry.module.train(train and entry.freeze is not True)
