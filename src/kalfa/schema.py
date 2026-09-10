from dataclasses import dataclass


@dataclass(frozen=True)
class Ref:
    type: str
    lego: bool = False
    table: str | None = None
    section: str | None = None


class Schema:
    sections = ("include", "plugins", "params", "alias", "seed", "device", "rng", "data", "model", "metrics", "losses",
                "optimizers", "training", "generate", "plots", "figures", "sweep", "record")
    required = ("data", "model", "losses", "training", "record")
    unresolved = ("include", "plugins", "params", "alias")
    builtin_variables = ("datetime",)
    short_calls = (("training", "turn"), ("training", "checkpoint"), ("data", "feed"), ("sweep", "strategy"),
                   ("device",), ("rng",))
    data = ("source", "filter", "split", "batch", "preprocessors", "drop", "fields", "feed")
    data_required = ("source", "split", "batch", "fields", "feed")
    filter = ("query", "sets")
    ratio_split = ("ratios", "seed")
    preprocessor = ("uri", "params", "sets")
    field = ("preprocessors", "target")
    model = ("templates", "models")
    definition_of_model = ("inputs", "outputs", "nodes", "optimizer", "init", "ema", "trainable", "weights")
    model_required = ("inputs", "outputs", "nodes")
    template = ("variables", "inputs", "outputs", "nodes")
    node = ("uri", "template", "model", "params", "inputs", "outputs", "init", "repeat")
    composite_forbidden = ("optimizer", "init", "ema", "trainable", "weights")
    init = ("patterns",)
    entry = ("uri", "params", "every", "sets", "output", "target")
    definition = ("sets", "every", "inputs", "output", "target", "width", "height")
    optimizer = ("uri", "params", "loss", "schedule")
    training_fixed = ("turn", "predicts", "targets", "epochs", "steps", "loss", "checkpoint", "report", "stop",
                      "rules")
    rule = ("name", "when", "set", "after")
    plot = ("uri", "params", "inputs", "sets", "width", "height")
    sweep = ("strategy", "space", "objective", "record")
    objective = ("monitor", "mode", "at")
    range = ("low", "high", "log", "int", "steps")
    refs = {
        "criterion": Ref("criterion", lego=True),
        "objective": Ref("objective", lego=True),
        "metric": Ref("metric", lego=True),
        "schedule": Ref("schedule", lego=True),
        "init": Ref("init", lego=True),
        "trigger": Ref("trigger", lego=True),
        "generate": Ref("generate", lego=True, section="generate"),
        "pre": Ref("pre", lego=True, table="preprocessors"),
        "preprocessor": Ref("preprocessor", lego=True, table="preprocessors"),
        "model": Ref("model", table="models"),
        "loss": Ref("loss", table="losses"),
        "field": Ref("field"),
        "column": Ref("column"),
        "wire": Ref("wire"),
        "history": Ref("history"),
        "data": Ref("data"),
    }

    @classmethod
    def ref(cls, type_name):
        return cls.refs.get(type_name, Ref(str(type_name)))

    @classmethod
    def lego_types(cls):
        return [name for name, ref in cls.refs.items() if ref.lego]
