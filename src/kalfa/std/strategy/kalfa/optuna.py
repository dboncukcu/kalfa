import math

from kalfa.registration import lego
from kalfa.std.strategy.base import Choices, Strategy


@lego("/strategy/kalfa/optuna", alias="optuna",
      description="trials points proposed by optuna (tpe or random sampler) from the objectives fed back; "
                  "local loop only, no --id")
class OptunaSearch(Strategy):
    deterministic = False

    def __init__(self, trials, seed=0, sampler="tpe"):
        self.trials = int(trials)
        self.seed = int(seed)
        self.sampler = sampler
        self.study = None

    def total(self, space):
        return self.trials

    def ensure_study(self, mode):
        if self.study is None:
            try:
                import optuna
            except ImportError as exception:
                raise ImportError("the optuna strategy needs optuna, which kalfa depends on; the environment "
                                  "is missing it, reinstall it with uv sync") from exception
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            if self.sampler == "random":
                sampler = optuna.samplers.RandomSampler(seed=self.seed)
            else:
                sampler = optuna.samplers.TPESampler(seed=self.seed)
            self.study = optuna.create_study(direction="minimize" if mode == "min" else "maximize", sampler=sampler)
        return self.study

    def ask(self, space, mode="min"):
        study = self.ensure_study(mode)
        trial = study.ask()
        point = {}
        for name, entry in space.items():
            if isinstance(entry, Choices):
                point[name] = trial.suggest_categorical(name, list(entry.values))
            elif entry.integer:
                point[name] = trial.suggest_int(name, int(entry.low), int(entry.high), log=entry.log)
            else:
                point[name] = trial.suggest_float(name, entry.low, entry.high, log=entry.log)
        return trial, point

    def tell(self, trial, value):
        import optuna

        if value is None or (isinstance(value, float) and math.isnan(value)):
            self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            self.study.tell(trial, float(value))
