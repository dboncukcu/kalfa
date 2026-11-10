from kalfa.commands.sweep.space import SPACE_HINT, space_values
from kalfa.commands.sweep.strategies import STRATEGY_HINT, Grid, strategy_of
from kalfa.core.check import SERIES_PREFIXES
from kalfa.core.context import train_of
from kalfa.core.errors import BuildError, error


SWEEP_KEYS = ("strategy", "space", "objective", "record")
OBJECTIVE_KEYS = ("watch", "mode", "at")
MODES = ("min", "max")
MOMENTS = ("best", "last")


def series_problem(tree, watch):
    names = set(tree.section("losses")) | set(tree.section("metrics"))
    prefixes = set(SERIES_PREFIXES)
    evaluate = train_of(tree).get("evaluate")
    if isinstance(evaluate, list):
        prefixes.update(str(name) for name in evaluate)
    prefix, _, rest = watch.partition("/")
    if not rest or prefix not in prefixes:
        return f"{watch!r} does not look like a series such as val/<metric>"
    if prefix in ("train", "val", "test") and rest.split("/")[0] not in names:
        known = ", ".join(sorted(names)) or "none"
        return f"{watch!r} names no loss or metric; the losses and metrics are {known}"
    return None


def check_space(tree, space, problems):
    try:
        dimensions = space_values(space)
    except BuildError as exc:
        problems.append(error("sweep_space", str(exc), "sweep.space", hint=SPACE_HINT))
        return None
    params = getattr(getattr(tree, "config", None), "params", None) or {}
    known = ", ".join(sorted(str(name) for name in params))
    for name in dimensions:
        if name not in params:
            problems.append(error("sweep_param", f"{name!r} is not in the params section; a swept name is a param the "
                                  f"config reads as ${name}$", "sweep.space",
                                  hint=f"the params are {known}" if known else "define it under params"))
    return dimensions


def check_strategy(raw, space, dimensions, problems):
    try:
        strategy = strategy_of(raw)
    except BuildError as exc:
        problems.append(error("sweep_strategy", str(exc), "sweep.strategy", hint=STRATEGY_HINT))
        return
    if dimensions is not None and isinstance(strategy, Grid):
        try:
            strategy.axes(space)
        except BuildError as exc:
            problems.append(error("sweep_space", str(exc), "sweep.space"))


def check_objective(tree, objective, problems):
    where = "sweep.objective"
    if not isinstance(objective, dict):
        problems.append(error("sweep_objective", "sweep.objective is a mapping {watch, mode, at}", where,
                              hint="write objective: {watch: val/<metric>, mode: min}"))
        return
    for key in objective:
        if key not in OBJECTIVE_KEYS:
            problems.append(error("sweep_objective", f"{key!r} is not an objective key; the keys are "
                                  + ", ".join(OBJECTIVE_KEYS), where))
    watch = objective.get("watch")
    if not isinstance(watch, str) or not watch:
        problems.append(error("sweep_objective", "the objective needs watch, the series that ranks the points", where,
                              hint="such as watch: val/rmse"))
    else:
        message = series_problem(tree, watch)
        if message:
            problems.append(error("sweep_objective", message, where))
    mode = objective.get("mode")
    if mode is not None and mode not in MODES:
        problems.append(error("sweep_objective", f"mode is min or max, got {mode!r}", where))
    at = objective.get("at")
    if at is not None and at not in MOMENTS:
        problems.append(error("sweep_objective", f"at is best or last, got {at!r}", where))


def check_sweep(tree, raw, problems):
    if not isinstance(raw, dict):
        problems.append(error("sweep_key", "sweep is a mapping with strategy, space, objective and record", "sweep"))
        return
    for key in raw:
        if key not in SWEEP_KEYS:
            problems.append(error("sweep_key", f"{key!r} is not a sweep key; the keys are " + ", ".join(SWEEP_KEYS),
                                  f"sweep.{key}"))
    dimensions = check_space(tree, raw.get("space"), problems)
    check_strategy(raw.get("strategy"), raw.get("space"), dimensions, problems)
    check_objective(tree, raw.get("objective"), problems)
    record = raw.get("record")
    if not isinstance(record, str) or not record.strip():
        problems.append(error("sweep_record", "the sweep needs record, the root directory its points go under",
                              "sweep.record", hint="write record: runs/<name>, or give --record"))
