import importlib
import inspect

from kalfa.commands.guide.calls import call_notes, lego_note, legos_in, param_lines, use_of
from kalfa.commands.guide.seen import default_of, patch_notes, patch_text, range_text, ref_meaning, ref_name
from kalfa.commands.guide.words import and_list, short, unique, value_text
from kalfa.core import aliases
from kalfa.core.build import SLOT_KEYS, criterion_function
from kalfa.core.check import is_criterion
from kalfa.core.protocols import Objective


TRIGGERS = ("after_epoch", "every", "plateau", "above", "below", "time_budget")


def criterion_notes(keys):
    notes = []
    if keys.get("pred") is not None:
        notes.append(f"pred: compares the output {value_text(keys['pred'])}")
    if keys.get("target") is not None:
        notes.append(f"target: against the batch field {value_text(keys['target'])}")
    on = keys.get("on")
    if on is not None:
        listed = on if isinstance(on, list) else [on]
        notes.append(f"on: runs on the {and_list(listed)} {'set' if len(listed) == 1 else 'sets'} only")
    if keys.get("every") is not None:
        notes.append(f"every: computed every {value_text(keys['every'])} turns")
    if keys.get("rescale") is True:
        notes.append("rescale: compares in the original units, after undoing the preprocessor of the target")
    elif keys.get("rescale") is False:
        notes.append("rescale: false compares the preprocessed values")
    if keys.get("sample_weight") is not None:
        notes.append(f"sample_weight: the loss of every row is multiplied by the batch field "
                     f"{value_text(keys['sample_weight'])} before it is averaged")
    return notes


def default_notes(seen, path, keys, metric):
    node = seen.node(path)
    if node is None or node.target is None or not is_criterion(node):
        return []
    targets = seen.targets()
    target = keys.get("target")
    if target is None and targets is not None and len(targets) == 1:
        target = targets[0]
    notes = []
    missing = [key for key in ("pred", "target") if keys.get(key) is None]
    model = seen.predicts()
    pred_text = None
    if model is not None:
        output = seen.first_output(model)
        pred_text = (f"the output of {model} (the predicts model)" if output == model
                     else f"{output} (the first output of the predicts model {model})")
    target_text = f"the target {target}" if target is not None and keys.get("target") is None else None
    if targets is None and keys.get("target") is None:
        target_text = "the only target of the data"
    if missing == ["pred", "target"] and pred_text and target_text:
        notes.append(f"pred and target are not written, so kalfa compares {pred_text} with {target_text}")
    elif "pred" in missing and pred_text:
        notes.append(f"pred is not written, so kalfa takes {pred_text}")
    elif "target" in missing and target_text:
        notes.append(f"target is not written, so kalfa takes {target_text}")
    if metric and keys.get("rescale") is None and isinstance(target, str):
        notes += rescale_notes(seen, target, node.name)
    return notes


def inverts(name):
    entry = aliases.entry_of(name)
    try:
        target = aliases.import_target(entry.target if entry is not None else name)
    except Exception:
        return None
    if not inspect.isclass(target):
        return None
    return hasattr(target, "inverse_transform") and bool(getattr(target, "rescales", True))


def rescale_notes(seen, target, name):
    targets = seen.mapping("data", "targets")
    if target not in targets:
        return []
    chain = targets[target]
    names = [use_of(item) for item in (chain if isinstance(chain, list) else [chain]) if item is not None]
    if not names:
        return [f"rescale stays off: {target} is kept raw"]
    if "label" in names:
        return []
    names = [item for item in names if item]
    answers = [inverts(item) for item in names]
    able = [item for item, answer in zip(names, answers) if answer]
    if able:
        return [f"rescale is on because {and_list(able)} can be inverted, so {name} speaks the original units of "
                f"{target}"]
    if answers and all(answer is False for answer in answers):
        return [f"rescale stays off: {and_list(names)} cannot be inverted"]
    return [f"rescale is on when a preprocessor of {target} can be inverted, so {name} speaks its original units"]


def kind_notes(seen, path, use):
    target = seen.target(path)
    if target is None or not use or aliases.entry_of(use) is not None:
        return []
    if inspect.isclass(target) and issubclass(target, Objective):
        return ["an objective: it reads the outputs, the batch and the run context itself"]
    if inspect.isroutine(target):
        if criterion_function(target):
            return ["a function of (pred, target), so it is the criterion itself and the other params are bound to it"]
        return ["a function without two required params: kalfa calls it and uses what it returns"]
    return []


def loss_notes(seen, key, value, keys, path):
    name = str(key)
    resolved = seen.mapping(*keys)
    notes = call_notes(seen, path, value, SLOT_KEYS["loss"]) + kind_notes(seen, path, use_of(value))
    notes += criterion_notes(resolved) + default_notes(seen, path, resolved, False)
    minimizers = seen.minimizers(name)
    if minimizers:
        notes.append(f"{and_list(minimizers)} {'minimizes' if len(minimizers) == 1 else 'minimize'} it, so it gives "
                     f"the gradients")
    elif any(reader.startswith("losses.") for reader in seen.readers(path)):
        notes.append("no optimizer minimizes it directly")
    elif seen.mapping("optimizers"):
        notes.append("no optimizer minimizes it at the start, so it only observes")
    for rule, written, item in seen.patches():
        if written.endswith(".minimizes") and ref_name(item, "losses") == name:
            notes.append(f"rule {rule} makes {written.split('.')[1]} minimize it")
    return notes


def metric_notes(seen, key, value, keys, path):
    resolved = seen.mapping(*keys)
    notes = call_notes(seen, path, value, SLOT_KEYS["metric"])
    return notes + criterion_notes(resolved) + default_notes(seen, path, resolved, True)


def over_note(over):
    items = over if isinstance(over, list) else [over]
    plain_models = [ref_name(item, "models") for item in items]
    if len(items) > 1 and all(plain_models) and all(item == f"@models.{name}" for item, name in
                                                    zip(items, plain_models)):
        return f"over: steps the parameters of the models {and_list(plain_models)}"
    parts = []
    for item in items:
        if isinstance(item, dict):
            own = ", ".join(f"{key} {value_text(value)}" for key, value in item.items() if key not in ("of", "match"))
            text = f"the parameters of {ref_meaning(item.get('of'))}"
            text += f" matching {item['match']}" if item.get("match") else ""
            parts.append(text + (f" ({own})" if own else ""))
        elif isinstance(item, str) and item.startswith("@"):
            parts.append(f"the parameters of {ref_meaning(item)}")
        else:
            parts.append(value_text(item))
    return f"over: steps {and_list(parts)}"


def optimizer_notes(seen, key, value, keys, path):
    resolved = seen.mapping(*keys)
    notes = call_notes(seen, path, value, SLOT_KEYS["optimizer"])
    if "over" in resolved:
        notes.append(over_note(resolved["over"]))
    loss = ref_name(resolved.get("minimizes"), "losses")
    if loss:
        notes.append(f"minimizes: steps on the loss {loss}")
    schedule = use_of(resolved.get("schedule"))
    if schedule:
        notes.append(f"schedule: multiplies the lr by {schedule} after every update")
        notes.append(lego_note(schedule))
        notes += param_lines(schedule, resolved.get("schedule"), lead="schedule.")
    return notes + patch_notes(seen, path)


def checkpoint_notes(checkpoint):
    if not isinstance(checkpoint, dict):
        return []
    notes = []
    if checkpoint.get("watch"):
        better = "higher" if checkpoint.get("mode") == "max" else "lower"
        notes.append(f"keeps best.pt whenever {checkpoint['watch']} improves ({better} is better)")
    text = "writes no last.pt" if checkpoint.get("last") is False else "writes last.pt every turn"
    if checkpoint.get("every"):
        text += f", turn_<k>.pt every {value_text(checkpoint['every'])} turns"
    return notes + [text + " and final.pt at the end"]


def train_notes(seen, key, value, keys, path):
    resolved = seen.value(*keys)
    if path == "train":
        return []
    if key == "epochs":
        return [f"trains {value_text(resolved)} turns, each one pass over the train set"]
    if key == "updates" and isinstance(resolved, dict):
        turn = f", {value_text(resolved['turn'])} per turn" if resolved.get("turn") else ""
        return [f"trains {value_text(resolved.get('total'))} updates{turn}"]
    if key == "step":
        return call_notes(seen, "train.step", value)
    if key == "predicts":
        if resolved is False:
            return ["false: no model predicts, for a run whose model makes no predictions, such as a DDPM denoiser"]
        model = ref_name(resolved, "models")
        tail = " and they fill the predictions table" if seen.tabled() else ", and this data class writes no table"
        return [f"the predicts model is {model}: the step computes its outputs for every batch{tail}"] if model else []
    if key == "evaluate" and isinstance(resolved, list):
        return [f"evaluates the {and_list(resolved)} {'set' if len(resolved) == 1 else 'sets'} after every turn"]
    if key == "checkpoint":
        return checkpoint_notes(resolved)
    if key == "report":
        weights = "best.pt" if resolved == "best" else "the weights of the last turn"
        return [f"the final predictions, plots and samples use {weights}"]
    if key == "track" and isinstance(resolved, list):
        return [f"writes {and_list(f'track/{item}' for item in resolved)} every turn"]
    if key == "log":
        return [lego_note(name, seen.found(name)) for name in unique(legos_in(value, True))]
    return []


def trigger_text(use, params, target):
    def get(name):
        found = params.get(name)
        return found if found is not None else default_of(target, name)

    if use == "after_epoch":
        return f"from turn {value_text(get('at'))}"
    if use == "every":
        return f"every {value_text(get('epochs'))} turns"
    if use == "plateau":
        text = f"after {value_text(get('patience'))} turns without {get('watch')} improving"
        text += f" by {value_text(get('min_delta'))}" if get("min_delta") else ""
        return text + (" (higher is better)" if get("mode") == "max" else "")
    if use in ("above", "below"):
        return f"when {get('watch')} is {use} {value_text(get('value'))}"
    if use == "time_budget":
        return f"once {value_text(get('minutes'))} minutes have passed"
    return f"when {use} fires" if use else "when its trigger fires"


def rule_sentence(rule, target):
    when, after = rule.get("when"), rule.get("after")
    use = use_of(when)
    if when is None:
        fires = f"fires on the turn after rule {after} fired" if after else "fires as soon as it is ready"
    else:
        fires = f"fires {trigger_text(use, when if isinstance(when, dict) else {}, target)}"
        fires = f"asked from the turn after rule {after} fired, it {fires}" if after else fires
    effects = [patch_text(path, value) for path, value in (rule.get("set") or {}).items()]
    if rule.get("stop"):
        effects.append("stops the run after that turn")
    if rule.get("run") is not None:
        effects.append(f"runs {use_of(rule['run']) or value_text(rule['run'])}")
    text = f"{fires} and {and_list(effects)}" if effects else f"{fires} but has no set, stop or run"
    if rule.get("once") is False:
        text += "; with once false it may fire again, and its trigger is reset after each firing"
    return text[0].upper() + text[1:] + "."


def rule_notes(seen, key, value, keys, path):
    resolved = seen.mapping(*keys)
    if not resolved:
        return []
    notes = [rule_sentence(resolved, seen.target(f"{path}.when"))]
    use = use_of(resolved.get("when"))
    if use and use not in TRIGGERS:
        notes += call_notes(seen, f"{path}.when", resolved.get("when"))
    if resolved.get("run") is not None:
        notes += call_notes(seen, f"{path}.run", resolved.get("run"))
    return notes


def plot_notes(seen, key, value, keys, path):
    notes = call_notes(seen, path, value)
    target = seen.target(path)
    if target is not None and not getattr(target, "__module__", "").startswith("kalfa.std"):
        notes.append(f"a figure it returns is saved as plots/{key}.png")
    return notes


def generate_notes(seen, key, value, keys, path):
    if path == "generate":
        return call_notes(seen, "generate", value)
    if key == "use":
        return [lego_note(value, seen.target("generate"))]
    entry = aliases.entry_of(use_of(seen.mapping("generate")) or "")
    note = entry.params.get(key) if entry is not None else None
    return [f"{key}: {short(note)}"] if note else []


def strategy_name(use):
    module = importlib.import_module("kalfa.commands.sweep")
    target = getattr(module, "STRATEGIES", {}).get(use)
    if target is None:
        return use
    path = aliases.target_path(target)
    names = [entry.name for entry in aliases.catalog(slot="strategy") if aliases.target_path(entry.target) == path]
    return names[0] if names else use


def point_count(seen):
    from kalfa.commands.sweep.strategies import strategy_of

    strategy = strategy_of(seen.mapping("sweep").get("strategy"))
    if callable(getattr(strategy, "points", None)):
        return len(strategy.points(seen.space(), 0))
    return getattr(strategy, "trials", None)


def sweep_notes(seen, key, value, keys, path):
    resolved = seen.value(*keys)
    if key == "strategy":
        use = use_of(resolved) or "grid"
        name = strategy_name(use)
        notes = [lego_note(name)] + param_lines(name, resolved)
        try:
            count = point_count(seen)
        except Exception:
            count = None
        return notes + ([f"the sweep runs {count} points"] if count else [])
    if key == "space" and isinstance(resolved, dict):
        notes = []
        for name, entry in resolved.items():
            paths = seen.uses.get(str(name), [])
            where = f", read as ${name}$ by {and_list(paths[:3])}" if paths else ""
            notes.append(f"{name}: {range_text(entry, seen.grid)}{where}")
        return notes
    if key == "objective" and isinstance(resolved, dict):
        better = "higher" if resolved.get("mode") == "max" else "lower"
        at = resolved.get("at") or "best"
        return [f"ranks the points by the {at} value of {resolved.get('watch')} over the turns of each point, "
                f"{better} being better"]
    if key == "record":
        return [f"the sweep root: the points are written to {value_text(resolved)}/0000, "
                f"{value_text(resolved)}/0001 and so on"]
    return []
