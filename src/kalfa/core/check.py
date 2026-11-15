import inspect
import os
from dataclasses import dataclass, field

import torch

from . import aliases, build, plan, spec
from .context import train_of
from .errors import ProblemsError, RunError, error, warning
from .protocols import ContextMetric, Objective
from .refs import SINGULAR, join_tokens, ref_parts, tokenize


TRAIN_KEYS = ("epochs", "updates", "step", "predicts", "evaluate", "checkpoint", "report", "track", "log")
CHECKPOINT_KEYS = ("watch", "mode", "last", "every")
SECTIONS = ("models", "losses", "metrics", "optimizers")
SET_NAMES = ("train", "valid", "test")
SERIES_PREFIXES = ("train", "val", "test", "track", "lr", "minimizes")
STD_STEPS = ("supervised", "alternating")


@dataclass
class Checked:
    config: object
    tree: object
    plan: dict | None
    problems: list = field(default_factory=list)

    @property
    def errors(self):
        return [problem for problem in self.problems if problem.severity == "error"]

    @property
    def warnings(self):
        return [problem for problem in self.problems if problem.severity == "warning"]

    @property
    def ok(self):
        return not self.errors

    def text(self):
        return "\n".join(str(problem) for problem in self.problems) or "ok"


def merge(into, more):
    for problem in more or ():
        if problem not in into:
            into.append(problem)
    return into


def has_errors(problems):
    return any(problem.severity == "error" for problem in problems)


def positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def ref_target(value):
    if isinstance(value, build.Ref):
        return value.path, value.attr
    text = ref_parts(value)
    if text is None:
        return None, None
    try:
        tokens = tokenize(text)
    except RunError:
        return None, None
    split = 2 if tokens[0] in SECTIONS else 1
    return join_tokens(tokens[:split]), join_tokens(tokens[split:]) or None


def ref_text(value):
    return value.text if isinstance(value, build.Ref) else value


def mutable_set(tree):
    found = build.mutable_paths(tree)
    if isinstance(found, dict):
        found = found.keys()
    return {str(path) for path in found or ()}


def nested_paths(prefix, value):
    paths = set()
    if isinstance(value, dict) and "use" not in value:
        for key, inner in value.items():
            leaf = f"{prefix}.{key}"
            paths.add(leaf)
            paths |= nested_paths(leaf, inner)
    elif isinstance(value, (list, tuple)):
        for index, inner in enumerate(value):
            leaf = f"{prefix}[{index}]"
            paths.add(leaf)
            paths |= nested_paths(leaf, inner)
    return paths


def known_paths(tree):
    paths = set()
    for node in tree.nodes.values():
        paths.add(node.path)
        paths.update(f"{node.path}.{key}" for key in build.SLOT_KEYS.get(node.slot, ()))
        for bucket in (node.params, node.keys):
            for key, value in (bucket or {}).items():
                leaf = f"{node.path}.{key}"
                paths.add(leaf)
                paths |= nested_paths(leaf, value)
    for key, value in train_of(tree).items():
        paths.add(f"train.{key}")
        paths |= nested_paths(f"train.{key}", value)
    return paths


def node_for(tree, path):
    try:
        tokens = tokenize(path)
    except RunError:
        return None, None
    for count in range(len(tokens), 0, -1):
        text = join_tokens(tokens[:count])
        if text in tree.nodes:
            return tree.nodes[text], tokens[count:]
    return None, tokens


def nearest(tree, node, mutable, path):
    if node is not None:
        under = sorted(item for item in mutable if item.startswith((f"{node.path}.", f"{node.path}[")))
        if under:
            return f"mutable under {node.path}: " + ", ".join(under)
        return f"nothing under {node.path} can be set by a rule"
    section = path.split(".", 1)[0].split("[", 1)[0]
    if section == "train":
        under = sorted(item for item in mutable if item.startswith("train."))
        return "mutable under train: " + ", ".join(under) if under else "nothing under train can be set by a rule"
    names = sorted(tree.section(section)) if section in SECTIONS else []
    if names:
        return f"the {section} are " + ", ".join(names)
    return "a path starts with " + ", ".join(SECTIONS + ("train",))


def check_set_path(tree, rule, path, mutable, known, problems):
    if path in mutable:
        return
    where = f"rules.{rule.name}.set"
    node, _ = node_for(tree, path)
    hint = nearest(tree, node, mutable, path)
    if path in known:
        problems.append(error("not_mutable", f"{path} is fixed and cannot be set by a rule", where, hint=hint))
    else:
        problems.append(error("unknown_path", f"{path} is not a path of the plan", where, hint=hint))


def check_set_value(tree, rule, path, value, problems):
    if ref_parts(value) is None:
        return
    where = f"rules.{rule.name}.set"
    target, _ = ref_target(value)
    if target is None:
        problems.append(error("rule_ref", f"{path} is set to {value!r}, which is not a reference", where))
        return
    if target not in tree.nodes:
        section = target.split(".", 1)[0]
        names = sorted(tree.section(section)) if section in SECTIONS else []
        hint = f"the {section} are " + ", ".join(names) if names else None
        problems.append(error("rule_ref", f"{path} is set to {value!r}, which names no node", where, hint=hint))


def check_cycles(specs, problems):
    after = {rule.name: rule.after for rule in specs}
    reported = set()
    for name in after:
        seen = []
        current = name
        while current is not None and current in after and current not in seen:
            seen.append(current)
            current = after[current]
        if current == name and tuple(sorted(seen)) not in reported:
            reported.add(tuple(sorted(seen)))
            problems.append(error("rule_after", "rules " + ", ".join(seen) + " wait for each other and can never fire",
                                  f"rules.{name}.after"))


def data_weights(tree):
    node = tree.nodes.get("data")
    weights = (node.params or {}).get("weights") if node is not None else None
    return [str(name) for name in weights] if isinstance(weights, dict) else []


def data_targets(tree):
    node = tree.nodes.get("data")
    if node is None:
        return False, []
    params = node.params or {}
    table = node.use == "table" or "targets" in params or getattr(node.target, "__name__", "") == "Table"
    targets = params.get("targets")
    names = [str(name) for name in targets] if isinstance(targets, dict) else []
    return table, names


def check_train(tree, problems):
    settings = train_of(tree)
    keys_hint = "the train keys are " + ", ".join(TRAIN_KEYS)
    for key in settings:
        if key not in TRAIN_KEYS:
            problems.append(error("train_key", f"unknown train key {key!r}", f"train.{key}", hint=keys_hint))
    epochs = settings.get("epochs")
    updates = settings.get("updates")
    if epochs is not None and updates is not None:
        problems.append(error("train_unit", "train takes epochs or updates, not both", "train"))
    elif epochs is None and updates is None:
        problems.append(error("train_unit", "train needs epochs or updates: {total, turn}", "train"))
    if epochs is not None and not positive_int(epochs):
        problems.append(error("train_unit", f"epochs is a positive integer, got {epochs!r}", "train.epochs"))
    if updates is not None:
        if not isinstance(updates, dict) or not positive_int(updates.get("total")):
            problems.append(error("train_unit", "updates is a mapping {total, turn} with a positive integer total",
                                  "train.updates"))
        else:
            for key in updates:
                if key not in ("total", "turn"):
                    problems.append(error("train_key", f"unknown updates key {key!r}", f"train.updates.{key}",
                                          hint="the updates keys are total and turn"))
            if "turn" in updates and not positive_int(updates["turn"]):
                problems.append(error("train_unit", f"updates.turn is a positive integer, got {updates['turn']!r}",
                                      "train.updates.turn"))
    report = settings.get("report", "last")
    checkpoint = settings.get("checkpoint")
    if report not in ("best", "last"):
        problems.append(error("train_report", f"report is best or last, got {report!r}", "train.report"))
    elif report == "best" and not (isinstance(checkpoint, dict) and checkpoint.get("watch")):
        problems.append(error("report_best", "report: best needs a checkpoint with a watch series", "train.report",
                              hint="add checkpoint: {watch: val/<metric>}"))
    if checkpoint is not None:
        check_checkpoint(checkpoint, problems)
    evaluate = settings.get("evaluate")
    if evaluate is not None:
        check_evaluate(tree, evaluate, problems)
    predicts = settings.get("predicts")
    if predicts is not None and predicts is not False:
        target, attr = ref_target(predicts)
        if target is None or attr or not target.startswith("models.") or target not in tree.nodes:
            names = sorted(tree.section("models"))
            problems.append(error("predicts_ref", f"predicts names a model like @models.<name>, got {predicts!r}",
                                  "train.predicts", hint="the models are " + ", ".join(names) if names else None))
    track = settings.get("track")
    if track is not None and (not isinstance(track, list) or not all(isinstance(item, str) for item in track)):
        problems.append(error("train_track", f"track is a list of strings, got {track!r}", "train.track"))
    elif track:
        for path in track:
            check_track_path(tree, path, problems)
    log = settings.get("log")
    entries = [log] if isinstance(log, (str, dict)) else log
    if log is not None and (not isinstance(entries, list)
                            or not all(isinstance(item, str) or (isinstance(item, dict) and "use" in item)
                                       for item in entries)):
        problems.append(error("train_log", f"log is a list of logger names or calls, got {log!r}", "train.log"))


def check_track_path(tree, path, problems):
    where = "train.track"
    try:
        tokens = tokenize(path)
    except RunError as failure:
        problems.append(error("track_path", str(failure), where))
        return
    root = tokens[0]
    if root == "train":
        return
    if root not in SECTIONS:
        problems.append(error("unknown_path", f"{path} is not a path of the plan", where,
                              hint="a track path starts with " + ", ".join(SECTIONS + ("train",))))
        return
    if len(tokens) < 3 or not isinstance(tokens[1], str) or f"{root}.{tokens[1]}" not in tree.nodes:
        names = sorted(tree.section(root))
        hint = f"the {root} are " + ", ".join(names) if names else f"there are no {root}"
        problems.append(error("unknown_path", f"{path} names no {SINGULAR[root]} of the plan and one of its "
                                              f"values, like {root}.<name>.<param>", where, hint=hint))


def check_checkpoint(checkpoint, problems):
    if not isinstance(checkpoint, dict):
        problems.append(error("checkpoint_value", "checkpoint is a mapping {watch, mode, last, every}",
                              "train.checkpoint"))
        return
    for key in checkpoint:
        if key not in CHECKPOINT_KEYS:
            problems.append(error("checkpoint_key", f"unknown checkpoint key {key!r}", f"train.checkpoint.{key}",
                                  hint="the checkpoint keys are " + ", ".join(CHECKPOINT_KEYS)))
    watch = checkpoint.get("watch")
    if watch is not None and (not isinstance(watch, str) or not watch):
        problems.append(error("checkpoint_watch", f"watch is a series name like val/rmse, got {watch!r}",
                              "train.checkpoint.watch"))
    mode = checkpoint.get("mode", "min")
    if mode not in ("min", "max"):
        problems.append(error("checkpoint_value", f"mode is min or max, got {mode!r}", "train.checkpoint.mode"))
    every = checkpoint.get("every")
    if every is not None and not positive_int(every):
        problems.append(error("checkpoint_value", f"every is a positive integer, got {every!r}",
                              "train.checkpoint.every"))
    last = checkpoint.get("last", True)
    if not isinstance(last, bool):
        problems.append(error("checkpoint_value", f"last is true or false, got {last!r}", "train.checkpoint.last"))


def check_evaluate(tree, evaluate, problems):
    if not isinstance(evaluate, list) or not all(isinstance(name, str) and name for name in evaluate):
        problems.append(error("evaluate_set", f"evaluate is a list of set names, got {evaluate!r}", "train.evaluate"))
        return
    if len(set(evaluate)) != len(evaluate):
        problems.append(error("evaluate_set", "evaluate names a set twice", "train.evaluate"))
    table, _ = data_targets(tree)
    unknown = [name for name in evaluate if name not in SET_NAMES]
    if table and unknown:
        problems.append(error("evaluate_set", "evaluate names " + ", ".join(unknown) + ", which a table does not make",
                              "train.evaluate", hint="the sets of a table are " + ", ".join(SET_NAMES)))


def step_is_std(tree):
    node = tree.nodes.get("train.step")
    if node is None or node.use in STD_STEPS:
        return True
    return getattr(node.target, "__module__", "").startswith("kalfa.std")


def check_optimizers(tree, problems):
    make = error if step_is_std(tree) else warning
    losses = sorted(tree.section("losses"))
    hint = "the losses are " + ", ".join(losses) if losses else "there are no losses"
    for node in tree.section("optimizers").values():
        minimizes = (node.keys or {}).get("minimizes", (node.params or {}).get("minimizes"))
        if minimizes is None:
            problems.append(make("minimizes_required", f"optimizer {node.name!r} names no loss to minimize", node.path,
                                 hint='add minimizes: "@losses.<name>"'))
            continue
        target, attr = ref_target(minimizes)
        if target is None or attr or not target.startswith("losses.") or target not in tree.nodes:
            problems.append(error("minimizes_ref", f"minimizes references a loss like @losses.<name>, got "
                                  f"{ref_text(minimizes)!r}", f"{node.path}.minimizes", hint=hint))


def is_criterion(node):
    target = node.target
    if target is None:
        return False
    if inspect.isclass(target) and issubclass(target, (Objective, ContextMetric)):
        return False
    return True


def output_names(tree):
    outputs = []
    for node in tree.section("models").values():
        names = (node.keys or {}).get("outputs")
        if isinstance(names, str):
            names = [names]
        if isinstance(names, (list, tuple)) and names:
            outputs.extend(str(name) for name in names)
        else:
            outputs.append(node.name)
    return outputs


def check_criteria(tree, problems):
    models = tree.section("models")
    predicts = train_of(tree).get("predicts")
    target, attr = ref_target(predicts) if predicts is not None else (None, None)
    predicts_known = (len(models) == 1 and predicts is not False) or (target in tree.nodes and not attr)
    outputs = output_names(tree)
    table, targets = data_targets(tree)
    weights = data_weights(tree)
    outputs_hint = "the outputs are " + ", ".join(outputs) if outputs else None
    for section in ("losses", "metrics"):
        kind = SINGULAR[section]
        for node in tree.section(section).values():
            if not is_criterion(node):
                if (node.keys or {}).get("sample_weight") is not None:
                    problems.append(error("weight_objective", f"{kind} {node.name!r} is an objective; sample_weight "
                                          "belongs to a criterion, a loss of (pred, target)",
                                          f"{node.path}.sample_weight",
                                          hint="an objective reads the weight field from the batch itself"))
                continue
            keys = node.keys or {}
            pred = keys.get("pred")
            target = keys.get("target")
            if pred is None and not predicts_known:
                problems.append(error("pred_required", f"{kind} {node.name!r} needs pred: <output> because no single "
                                      "model predicts", node.path,
                                      hint=outputs_hint or "set train.predicts to the model whose output it compares"))
            elif pred is not None and outputs and pred not in outputs:
                problems.append(warning("unknown_output", f"pred {pred!r} is no output of a model", f"{node.path}.pred",
                                        hint=outputs_hint))
            if table and target is None and len(targets) != 1:
                hint = "the targets are " + ", ".join(targets) if targets else "the data declares no targets"
                problems.append(error("target_required", f"{kind} {node.name!r} needs target: <name> because the data "
                                      f"has {len(targets)} targets", node.path, hint=hint))
            weight = keys.get("sample_weight")
            if table and weight is not None and weight not in weights:
                hint = ("the weights are " + ", ".join(weights) if weights
                        else 'declare it under data.weights, such as weights: {w: {"price < 5": 5}}')
                problems.append(error("unknown_weight", f"sample_weight {weight!r} is no weight of the data",
                                      f"{node.path}.sample_weight", hint=hint))


def check_device(tree, problems):
    data = getattr(getattr(tree, "config", None), "data", None)
    device = data.get("device") if isinstance(data, dict) else None
    if device is None or device == "auto":
        return
    if not isinstance(device, str):
        problems.append(error("device_value", f"device is a string like cpu, cuda:0 or auto, got {device!r}", "device"))
        return
    try:
        torch.device(device)
    except (RuntimeError, TypeError, ValueError):
        problems.append(error("device_value", f"{device!r} is not a device torch knows", "device",
                              hint="use auto, cpu, mps, cuda or cuda:<index>"))


def check_series(text, names, prefixes, where, problems, strict=False):
    make = error if strict else warning
    code = "watch_unknown" if strict else "unknown_series"
    prefix, _, rest = text.partition("/")
    if not rest or prefix not in prefixes:
        problems.append(make(code, f"{text!r} does not look like a series such as val/<metric>", where))
        return
    if prefix in ("train", "val", "test") and rest.split("/")[0] not in names:
        problems.append(make(code, f"{text!r} names no loss or metric", where,
                             hint="the names are " + ", ".join(sorted(names)) if names else None))


def check_watches(tree, problems):
    names = set(tree.section("losses")) | set(tree.section("metrics"))
    settings = train_of(tree)
    prefixes = set(SERIES_PREFIXES)
    evaluate = settings.get("evaluate")
    if isinstance(evaluate, list):
        prefixes.update(str(name) for name in evaluate)
    checkpoint = settings.get("checkpoint")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("watch"), str) and checkpoint["watch"]:
        check_series(checkpoint["watch"], names, prefixes, "train.checkpoint.watch", problems,
                     strict=settings.get("report") == "best")
    for node in tree.nodes.values():
        watch = (node.params or {}).get("watch")
        if node.path.endswith(".when") and isinstance(watch, str) and watch:
            check_series(watch, names, prefixes, f"{node.path}.watch", problems)


def check_names(tree, problems):
    shared = set(tree.section("losses")) & set(tree.section("metrics"))
    for name in sorted(shared):
        problems.append(warning("series_clash", f"loss {name!r} and metric {name!r} write the same series "
                                f"train/{name}, val/{name} and test/{name}; the metric value is kept",
                                f"metrics.{name}", hint="rename one of them to record both"))


def check_data(tree, problems):
    node = tree.nodes.get("data")
    if node is None or node.target is None:
        return
    checker = inspect.getattr_static(node.target, "check", None)
    if not isinstance(checker, (classmethod, staticmethod)):
        return
    directories = getattr(getattr(tree, "config", None), "directories", None) or ()
    node.target.check(dict(node.params or {}), problems, directories)


def check_data_refs(tree, problems):
    node = tree.nodes.get("data")
    exposes = getattr(node.target, "exposes", None) if node is not None and node.target is not None else None
    if exposes is None:
        return
    exposes = tuple(str(name) for name in exposes)
    for owner in tree.nodes.values():
        for ref in owner.refs:
            if ref.path != "data" or not ref.attr:
                continue
            first = ref.attr.split(".")[0]
            if first not in exposes:
                problems.append(error("reference_unknown", f"{ref.at}: {ref.text} is not exposed", owner.path,
                                      hint="the data class exposes " + ", ".join(exposes)))


def check_actions(tree, problems):
    has_generate = "generate" in tree.nodes
    for node in tree.nodes.values():
        if node.slot != "action" or not getattr(node.target, "requires_generate", False):
            continue
        if node.params.get("sampler") is None and not has_generate:
            problems.append(error("generate_required", f"{node.use} needs a sampler or a generate section", node.path,
                                  hint="write sampler: {use: ...} on the action or a generate section"))


def check_sweep(tree, problems):
    raw = tree.settings.get("sweep") if isinstance(tree.settings, dict) else None
    if raw is None:
        return
    from kalfa.commands.sweep import check_sweep as checker

    checker(tree, raw, problems)


def validate(tree):
    problems = []
    specs = tree.rules
    mutable = mutable_set(tree)
    known = known_paths(tree)
    for rule in specs:
        for path, value in rule.set.items():
            check_set_path(tree, rule, path, mutable, known, problems)
            check_set_value(tree, rule, path, value, problems)
    check_cycles(specs, problems)
    check_data(tree, problems)
    check_data_refs(tree, problems)
    check_train(tree, problems)
    check_optimizers(tree, problems)
    check_criteria(tree, problems)
    check_device(tree, problems)
    check_watches(tree, problems)
    check_names(tree, problems)
    check_actions(tree, problems)
    check_sweep(tree, problems)
    return problems


def check(paths=(), sets=(), params=(), mappings=()):
    config = spec.load(paths=paths, sets=sets, params=params, mappings=mappings)
    problems = merge([], config.problems)
    for directory in config.directories:
        aliases.add_search_dir(directory)
    aliases.add_search_dir(os.getcwd())
    if has_errors(problems):
        return Checked(config, None, None, problems)
    aliases.import_plugins(config.plugins, problems)
    if has_errors(problems):
        return Checked(config, None, None, problems)
    try:
        tree = build.build(config)
    except ProblemsError as failure:
        return Checked(config, None, None, merge(problems, failure.problems))
    merge(problems, tree.problems)
    merge(problems, validate(tree))
    made = None
    if not has_errors(problems):
        from kalfa import __version__

        made = plan.plan_of(tree, __version__)
    return Checked(config, tree, made, problems)
