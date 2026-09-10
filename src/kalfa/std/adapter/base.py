from kalfa.kinds import SETS


def wires(keys):
    """The output wire and the target field a definition names; both None when it names nothing."""
    keys = keys or {}
    return keys.get("output"), keys.get("target")


def observed(value):
    """A tensor as an observation: off the graph. Metrics never backpropagate, and a metric that accumulates a
    tensor carrying gradients keeps the graph of every batch alive."""
    detach = getattr(value, "detach", None)
    return detach() if callable(detach) else value


def all_sets():
    return list(SETS)
