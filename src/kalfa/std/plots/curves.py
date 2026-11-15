from kalfa.records import reader
from kalfa.std.plots.common import LEGEND_LIMIT, color_of, directory_of, dress, quiet, spans_decades


DEFAULT_PREFIXES = ("train/", "val/")


def turn_word(directory):
    manifest = quiet(reader.read_manifest, directory) or {}
    return "epoch" if manifest.get("unit") == "epochs" else "turn"


def history_series(history, names=None):
    found = {}
    for position, line in enumerate(history):
        turn = reader.turn_of(line, position)
        for key, value in line.items():
            if not isinstance(key, str) or not reader.is_number(value):
                continue
            if names is None and not key.startswith(DEFAULT_PREFIXES):
                continue
            if names is not None and key not in names:
                continue
            turns, values = found.setdefault(key, ([], []))
            turns.append(turn)
            values.append(value)
    if names is not None:
        return {name: found[name] for name in names if name in found}
    return found


def loss_curve(record, figures, series=None):
    from matplotlib.ticker import MaxNLocator

    directory = directory_of(record)
    history = reader.read_history(directory)
    names = None if series is None else ([series] if isinstance(series, str) else [str(name) for name in series])
    found = history_series(history, names)
    if not found:
        return None
    figure = figures.new(8.0, 5.0)
    axis = figure.subplots()
    for position, (name, (turns, values)) in enumerate(found.items()):
        axis.plot(turns, values, label=name, color=color_of(figures, position), linewidth=1.8)
    if spans_decades([value for _, values in found.values() for value in values]):
        axis.set_yscale("log")
    if len(found) <= LEGEND_LIMIT:
        axis.legend(loc="upper right", frameon=False, ncols=1 if len(found) < 6 else 2)
    word = turn_word(directory)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_title(f"Training history ({len(history)} {word}s, {len(found)} series)", loc="left")
    axis.set_xlabel(word)
    axis.set_ylabel("value")
    dress(axis)
    figures.save(figure, "loss_curve")
    return None
