import logging
from pathlib import Path

from kalfa.commands.sweep.checks import (
    MODES,
    MOMENTS,
    OBJECTIVE_KEYS,
    SWEEP_KEYS,
    check_objective,
    check_space,
    check_strategy,
    check_sweep,
    series_problem,
)
from kalfa.commands.sweep.plan import (
    MAPPING_NOTE,
    Plan,
    command_args,
    condor_arg,
    executable,
    objective_note,
    plan_of,
    plan_seed,
    points_text,
    previous_note,
    script_text,
    submit_text,
    value_text,
    write_plan,
    write_text,
)
from kalfa.commands.sweep.points import (
    LOGS,
    OUTSIDE_STOPS,
    POINTS_FILE,
    REPAIRED,
    STOP_FILE,
    SWEEP_FILE,
    abandoned,
    append_point,
    clear_point,
    complete,
    failed_points,
    manifest_of,
    next_index,
    planned_values,
    point_dir,
    recorded_points,
    recorded_values,
    repairable,
    stopped_by_rule,
)
from kalfa.commands.sweep.runner import (
    LOGGER,
    child,
    in_children,
    interrupted,
    resumed,
    run_point,
    say_of,
    silent,
    values_text,
)
from kalfa.commands.sweep.space import (
    RANGES,
    SPACE_HINT,
    bounds_of,
    dimension,
    draw,
    is_int,
    space_values,
    spread,
)
from kalfa.commands.sweep.strategies import (
    OPTUNA_MISSING,
    SAMPLERS,
    STRATEGIES,
    STRATEGY_HINT,
    Grid,
    Optuna,
    Random,
    distribution_of,
    finished_trial,
    is_plain,
    optuna_module,
    resumed_seed,
    seed_value,
    strategy_call,
    strategy_names,
    strategy_note,
    strategy_of,
    strategy_target,
    suggest,
)
from kalfa.core import aliases
from kalfa.core.errors import KalfaError
from kalfa.records import reader
from kalfa.records.sweep import objective_of
from kalfa.std.common import positive_int as positive


__all__ = ["LOGGER", "LOGS", "MAPPING_NOTE", "MODES", "MOMENTS", "OBJECTIVE_KEYS", "OPTUNA_MISSING", "OUTSIDE_STOPS",
           "POINTS_FILE", "RANGES", "REPAIRED", "SAMPLERS", "SPACE_HINT", "STOP_FILE", "STRATEGIES", "STRATEGY_HINT",
           "SWEEP_FILE", "SWEEP_KEYS", "Grid", "Optuna", "Plan", "Random", "abandoned", "append_point", "asking",
           "attempt", "bounds_of", "check_objective", "check_space", "check_strategy", "check_sweep", "child",
           "clear_point", "command_args", "complete", "condor_arg", "dimension", "distribution_of", "draw",
           "executable", "failed_points", "finished_trial", "in_children", "in_turn", "interrupted", "is_int",
           "is_plain", "manifest_of", "next_index", "objective_note", "optuna_module", "plan_of", "plan_seed",
           "planned_values", "point_dir", "points_text", "positive", "previous_note", "recorded_points",
           "recorded_values", "repairable", "resumed", "resumed_seed", "run_local", "run_point", "say_of",
           "script_text", "seed_value", "series_problem", "silent", "space_values", "spread", "stopped_by_rule",
           "strategy_call", "strategy_names", "strategy_note", "strategy_of", "strategy_target", "submit_text",
           "suggest", "value_text", "values_text", "write_plan", "write_text"]


def attempt(plan, index, values, monitor):
    try:
        values = values if values is not None else planned_values(plan, index)
        say_of(monitor)(f"point {index:04d}  {values_text(values)}")
        return run_point(plan, index, values, monitor)
    except Exception as exc:
        LOGGER.warning("point %04d failed: %s: %s", index, type(exc).__name__, exc,
                       exc_info=LOGGER.isEnabledFor(logging.DEBUG))
        return None


def in_turn(plan, indices, monitor, log):
    results = []
    for index in indices:
        result = attempt(plan, index, None, monitor)
        results.append(result)
        if interrupted(result):
            log(f"point {index:04d} was stopped ({result.context.stopping}); the sweep stops here and --repair runs "
                f"the rest")
            break
    return results


def asking(plan, monitor, log):
    trials = int(plan.strategy.trials)
    start = next_index(plan)
    if start >= trials:
        log(f"{plan.root}: the {trials} trials have run")
        return []
    if start:
        told = resumed(plan)
        if told is None:
            log(f"{plan.root}: {start} points exist; trials go on from point {start:04d} and the strategy, which has "
                f"no resume(space, objective, points), does not know the points already run")
        else:
            log(f"{plan.root}: {start} points exist and the strategy was told {told} of them; trials go on from point "
                f"{start:04d}")
    results = []
    for index in range(start, trials):
        trial, values = plan.strategy.ask(plan.space, plan.objective)
        result = attempt(plan, index, values, monitor)
        value = None if result is None else objective_of(reader.summary_of(result.record), plan.objective)
        plan.strategy.tell(trial, value)
        append_point(plan.root, index, values)
        results.append(result)
        if interrupted(result):
            log(f"point {index:04d} was stopped ({result.context.stopping}); the sweep stops here")
            break
    return results


def run_local(plan, monitor=None, jobs=1, repair=False, log=print):
    jobs = max(1, int(jobs or 1))
    if jobs > 1 and not plan.paths:
        raise KalfaError("jobs above 1 run the points in child processes, which read config files; a mapping config "
                         "runs with jobs 1")
    if not (Path(plan.root) / SWEEP_FILE).exists():
        write_plan(plan, log)
    if repair:
        indices = failed_points(plan)
        total = plan.count if plan.count is not None else len(recorded_values(plan))
        log(f"{plan.root}: {len(indices)} of {total} points to repair"
            + (": " + ", ".join(f"{index:04d}" for index in indices) if indices else ""))
    elif plan.adaptive:
        return asking(plan, monitor, log)
    else:
        indices = []
        for index in range(len(plan.points)):
            directory = Path(point_dir(plan.root, index))
            if complete(directory):
                log(f"point {index:04d} is {manifest_of(directory).get('status')}, skipped")
            else:
                indices.append(index)
    if jobs > 1:
        return in_children(plan, indices, jobs, log)
    return in_turn(plan, indices, monitor, log)


Grid.__module__ = Optuna.__module__ = Random.__module__ = __name__

aliases.register("grid", "kalfa.commands.sweep.Grid", slot="strategy",
                 doc="every combination of the space; a range entry needs steps")
aliases.register("optuna", "kalfa.commands.sweep.Optuna", slot="strategy",
                 doc="trials points asked from an optuna study (tpe or random sampler), one after another")
aliases.register("random_search", "kalfa.commands.sweep.Random", slot="strategy",
                 doc="trials points drawn at random over the space with numpy's default_rng(seed), repeats drawn "
                     "again; a sweep section also takes the short name random")
