FACTS = ("uses", "needs_grad", "needs_models", "extras", "grouped", "requires")

KINDS = ("source", "split", "pre", "feed", "loader", "layer", "init", "criterion", "objective", "metric",
         "adapter", "optimizer", "schedule", "turn", "trigger", "checkpoint", "rule", "generate", "plot", "strategy",
         "device", "lego", "builder", "data")

CIRAK_KINDS = {"trigger": "predicate"}


def kind_of(uri):
    if not isinstance(uri, str) or not uri.startswith("/") or uri.count("/") < 2:
        raise ValueError(f"{uri!r} is not a lego URI (/<kind>/<pack>/<name>)")
    kind = uri.split("/", 2)[1]
    if kind not in KINDS:
        raise ValueError(f"{uri}: {kind!r} is not a kalfa kind; the kinds are {list(KINDS)}")
    return kind


def names_of(value):
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def cirak_kind(kind):
    return CIRAK_KINDS.get(kind, kind)


def kalfa_kind(uri):
    try:
        return kind_of(uri)
    except ValueError:
        return None

SETS = ("train", "valid", "test")

HISTORY_PREFIX = {"train": "train", "valid": "val", "test": "test"}

RESERVED_BLOCKS = ("data", "models", "optimizers", "training", "after")

TRAINING_FIXED = ("turn", "predicts", "targets", "epochs", "steps", "loss", "checkpoint", "report", "stop",
                  "rules")

DEFINITION_KEYS = ("sets", "every", "inputs", "output", "target", "width", "height")
