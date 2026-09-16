from kalfa.std.common.runtime import call_model
from kalfa.std.layer.kalfa.multipliers import Multipliers


SPEC_KEYS = ("epsilon", "lmbda_init", "scale", "damping")


def term_of(losses, name):
    head, _, term = name.partition(".")
    value = losses[head]
    if not term:
        return value["loss"] if isinstance(value, dict) else value
    if isinstance(value, dict) and term in value:
        return value[term]
    have = f"its terms are {sorted(value)}" if isinstance(value, dict) else "it returns one value"
    raise KeyError(f"losses definition {head!r} has no term {term!r}; {have}")


def constraint_specs(constraints):
    if not isinstance(constraints, dict) or not constraints:
        raise ValueError("mdmm needs constraints: a mapping of a losses name (or name.term) to {epsilon, lmbda_init, "
                         "scale, damping}")
    specs = {}
    for name, spec in constraints.items():
        spec = dict(spec) if isinstance(spec, dict) else {"epsilon": spec}
        if "epsilon" not in spec:
            raise ValueError(f"mdmm constraint {name!r} needs epsilon, the value the term is held at")
        unknown = sorted(set(spec) - set(SPEC_KEYS))
        if unknown:
            raise ValueError(f"mdmm constraint {name!r} has no {unknown}; a constraint writes {list(SPEC_KEYS)}")
        specs[str(name)] = spec
    return specs


def check_multipliers(model, name, names):
    found = next((module for module in model.modules() if isinstance(module, Multipliers)), None)
    if found is not None and found.names != names:
        raise ValueError(f"mdmm: model {name!r} holds multipliers for {found.names}, the constraints are {names}; "
                         f"write names: $constraints$ on the multipliers node so the two stay in step")


def mdmm(models, batch, losses, primary, multipliers, constraints):
    specs = constraint_specs(constraints)
    if multipliers not in models:
        raise KeyError(f"mdmm: multipliers names {multipliers!r}, which is no model; the models are {sorted(models)}")
    model = models[multipliers]
    check_multipliers(model, multipliers, list(specs))
    lmbda = call_model(model, batch).reshape(-1)
    if lmbda.numel() != len(specs):
        raise ValueError(f"mdmm: model {multipliers!r} returns {lmbda.numel()} multipliers for {len(specs)} "
                         f"constraints")
    value = term_of(losses, primary)
    out = {"loss": None, "primary": value}
    for position, (name, spec) in enumerate(specs.items()):
        infeasibility = float(spec["epsilon"]) - term_of(losses, name)
        damping = float(spec.get("damping", 1.0))
        value = value + float(spec.get("scale", 1.0)) * (lmbda[position] * infeasibility
                                                         + damping * infeasibility ** 2 / 2)
        out[f"lambda/{name}"] = lmbda[position]
        out[f"inf/{name}"] = infeasibility
    out["loss"] = value
    return out
