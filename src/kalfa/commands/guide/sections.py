from kalfa.commands.guide.about import DATA_KEYS, RUN_NOTES
from kalfa.commands.guide.calls import COLUMNS, LEGO_KEYS, call_notes, lego_note, legos_in, param_lines, use_of
from kalfa.commands.guide.seen import (layer_notes, override_notes, param_value_notes, patch_notes, ref_meaning,
                                       ref_name, refs_of, swept)
from kalfa.commands.guide.words import and_list, short, unique, value_text
from kalfa.core import aliases
from kalfa.core.build import SLOT_KEYS
from kalfa.core.plan import sets_of


QUIET_KEYS = {"optimizers": ("over", "minimizes"), "rules": ("set",)}
PREDICTS = "its outputs fill the predictions table"
UNTABLED = "the step computes its outputs for every batch, and this data class writes no predictions table"


def series_note(seen, section, path):
    node = seen.node(path) if section in ("losses", "metrics", "optimizers") else None
    if node is None:
        return []
    if section == "optimizers":
        return [f"series lr/{node.name} and minimizes/{node.name}"]
    series = [f"{prefix}/{node.name}" for prefix in sets_of(node)]
    series = ([f"train/{node.name}"] if section == "losses" else []) + series
    return [f"series {and_list(unique(series))}"]


def common_notes(seen, section, value, path, keys):
    hidden = QUIET_KEYS.get(section, ())
    shown = {key: item for key, item in value.items() if key not in hidden} if isinstance(value, dict) else value
    shown = None if path == "train.predicts" else shown
    notes = [f"{text} is {ref_meaning(text)}" for text in unique(text for _, text in refs_of(shown))]
    notes += param_value_notes(seen, value)
    readers = seen.readers(path) if section in ("models", "losses", "metrics") else []
    notes += [f"read as @{path} by {and_list(readers)}"] if readers else []
    notes += patch_notes(seen, path) if section not in ("optimizers", "rules") else []
    notes += series_note(seen, section, path)
    return notes + layer_notes(seen, keys, value) + override_notes(seen, path)


def run_notes(seen, key, value, keys, path):
    if key == "device":
        device = seen.value("device")
        if device in (None, "auto"):
            return ["auto takes cuda when it is there, then mps, then cpu"]
        return [f"the run trains on {value_text(device)}"]
    if key == "include":
        return ["merges these files below this one, so a key written here wins over theirs"]
    if key == "plugins":
        modules = getattr(seen.config, "plugins", None) or []
        return [f"imported before the build: {and_list(modules)}; the names they register with kalfa.lego become "
                "short names"]
    return [RUN_NOTES[key]] if key in RUN_NOTES else []


def param_notes(seen, key, value, keys, path):
    paths = seen.uses.get(str(key), [])
    if paths:
        shown = paths[:5] + ([f"{len(paths) - 5} more"] if len(paths) > 5 else [])
        notes = [f"read as ${key}$ by {and_list(shown)}"]
    else:
        notes = [f"nothing reads ${key}$"]
    sweep = swept(seen, str(key))
    return notes + ([sweep] if sweep else [])


def split_text(ratios):
    if not ratios or not all(isinstance(ratio, (int, float)) and not isinstance(ratio, bool) for ratio in ratios):
        return None
    names = ("train", "valid", "test")
    shares = [f"{ratio * 100:g}% {name}" for name, ratio in zip(names, ratios) if ratio]
    missing = [name for index, name in enumerate(names) if index >= len(ratios) or not ratios[index]]
    tail = f", with no {and_list(missing)} set" if missing else ""
    return f"a shuffled split: {and_list(shares)}{tail}, seeded by the top level seed"


def target_notes(targets):
    notes = []
    for name, chain in (targets or {}).items():
        chain = chain if isinstance(chain, list) else [chain]
        names = [use_of(item) for item in chain if item is not None]
        if not names:
            notes.append(f"{name} is kept raw")
        elif "label" in names:
            notes.append(f"{name} is a classification target: label gives int64 class codes")
        else:
            notes.append(f"{name} is preprocessed; predictions and rescaled metrics come back in its original units")
    return notes


def data_notes(seen, key, value, keys, path):
    if path == "data":
        return call_notes(seen, "data", value)
    target = seen.target("data")
    if key == "use":
        notes = [lego_note(value, target)]
        inputs = getattr(target, "inputs", None)
        exposes = getattr(target, "exposes", None)
        if inputs:
            notes.append(f"a model takes {and_list(inputs)} as inputs when it names no inputs")
        if exposes:
            notes.append(f"@data reads {and_list(exposes)} once it is fitted")
        return notes
    use = use_of(seen.mapping("data"))
    entry = aliases.entry_of(use) if use else None
    note = (entry.params.get(key) if entry is not None else None) or DATA_KEYS.get(key)
    notes = [f"{key}: {short(note)}"] if note else []
    resolved = seen.value(*keys)
    if key == "split" and isinstance(resolved, list):
        notes.append(split_text(resolved))
    if key == "targets" and isinstance(resolved, dict):
        notes += target_notes(resolved)
    names = unique(legos_in(value, key in LEGO_KEYS or key in COLUMNS))
    inner = use_of(value) if key in LEGO_KEYS and isinstance(value, dict) else None
    if inner:
        notes += [lego_note(inner)] + param_lines(inner, value)
    return notes + [lego_note(name, seen.found(name)) for name in names if name != inner]


def model_key_notes(name, keys):
    notes = []
    inputs, outputs = keys.get("inputs"), keys.get("outputs")
    if inputs is not None:
        listed = inputs if isinstance(inputs, list) else [inputs]
        notes.append(f"inputs: called with the batch fields {and_list(listed)} in this order")
    if outputs is not None:
        listed = outputs if isinstance(outputs, list) else [outputs]
        notes.append(f"outputs: names what it returns {and_list(listed)}; a tensor takes the first name and a tuple "
                     "matches them in order")
    else:
        notes.append(f"without outputs, a tensor it returns is named {name}")
    ema = keys.get("ema")
    if ema is not None and ema is not False:
        decay = ema.get("decay") if isinstance(ema, dict) else ema
        every = ema.get("every") if isinstance(ema, dict) else None
        text = "ema: keeps an exponential moving average copy"
        text += f" with decay {value_text(decay)}" if decay is not None and decay is not True else ""
        text += f", updated every {value_text(every)} steps" if every else ""
        notes.append(text + f", read as @models.{name}.ema")
    freeze = keys.get("freeze")
    if freeze is True:
        notes.append("freeze: every parameter is frozen and the module stays in eval mode")
    elif freeze is False:
        notes.append("freeze: false, nothing is frozen; a rule may patch it")
    elif isinstance(freeze, (list, str)):
        listed = freeze if isinstance(freeze, list) else [freeze]
        notes.append(f"freeze: the parameters matching {and_list(listed)} are frozen")
    weights = keys.get("weights")
    if isinstance(weights, dict):
        source = f" from the record {weights['run']}" if weights.get("run") else ""
        notes.append(f"weights: loads the {weights.get('which') or 'last'} checkpoint of the model "
                     f"{weights.get('model') or name}{source}")
    elif isinstance(weights, str):
        notes.append(f"weights: loads the last checkpoint of the model {name} from the record {weights}")
    init = use_of(keys.get("init"))
    if init:
        notes += [f"init: applies {init} to the module once it is built", lego_note(init)]
        notes += param_lines(init, keys.get("init"), lead="init.")
    return notes


def model_notes(seen, key, value, keys, path):
    name = str(key)
    notes = call_notes(seen, path, value, SLOT_KEYS["model"])
    notes += model_key_notes(name, seen.mapping(*keys))
    if seen.predicts() == name:
        written = seen.mapping("train").get("predicts") is not None
        how = "train.predicts names it" if written else "the only model, so it is the predicts model"
        notes.append(f"{how}: {PREDICTS if seen.tabled() else UNTABLED}")
    steppers = [opt for opt, item in seen.mapping("optimizers").items()
                if isinstance(item, dict) and name in over_models(item.get("over"))]
    if steppers:
        notes.append(f"{and_list(steppers)} {'steps' if len(steppers) == 1 else 'step'} its parameters")
    elif seen.mapping("optimizers"):
        notes.append("no optimizer names it in over")
    return notes


def over_models(over):
    items = over if isinstance(over, list) else [over]
    found = []
    for item in items:
        ref = item.get("of") if isinstance(item, dict) else item
        name = ref_name(ref, "models")
        if name:
            found.append(name)
    return found
