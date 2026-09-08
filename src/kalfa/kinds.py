"""The lego kinds and facts kalfa declares to cirak, and where each kind may be written in a config."""

FACTS = ("uses", "needs_grad", "needs_models", "extras", "grouped")

KINDS = ("source", "split", "pre", "feed", "loader", "layer", "init", "criterion", "objective", "metric",
         "adapter", "optimizer", "schedule", "turn", "trigger", "checkpoint", "rule", "generate", "plot", "strategy",
         "device", "lego", "builder", "data")

CIRAK_KINDS = {"trigger": "predicate"}


def kind_of(uri):
    """The kind of a lego: the first segment of its URI, one of KINDS."""
    if not isinstance(uri, str) or not uri.startswith("/") or uri.count("/") < 2:
        raise ValueError(f"{uri!r} is not a lego URI (/<kind>/<pack>/<name>)")
    kind = uri.split("/", 2)[1]
    if kind not in KINDS:
        raise ValueError(f"{uri}: {kind!r} is not a kalfa kind; the kinds are {list(KINDS)}")
    return kind


def names_of(value):
    """A fact written as a name or a list of names, as a tuple; cirak stores a declared fact as it was written."""
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def cirak_kind(kind):
    """The kind name cirak stores for a kalfa kind: triggers are cirak predicates, the rest keep their names."""
    return CIRAK_KINDS.get(kind, kind)


def kalfa_kind(uri):
    """The kalfa kind of a registered URI, None when the URI does not follow the rule (a foreign registration)."""
    try:
        return kind_of(uri)
    except ValueError:
        return None

SETS = ("train", "valid", "test")

HISTORY_PREFIX = {"train": "train", "valid": "val", "test": "test"}

RESERVED_BLOCKS = ("data", "models", "optimizers", "training", "after")

TRAINING_FIXED = ("turn", "predicts", "targets", "epochs", "steps", "loss", "checkpoint", "report", "stop",
                  "rules")

DEFINITION_KEYS = ("sets", "every", "inputs", "output", "target")
