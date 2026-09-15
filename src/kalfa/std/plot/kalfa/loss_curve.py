from kalfa.std.common.figure import Figure
from kalfa.std.common.history import History
from kalfa.std.plot.base import turn_word


def draw_series(figures, axis, found, positions, log):
    for label, values in found.items():
        axis.plot(positions[:len(values)] if len(positions) >= len(values) else range(1, len(values) + 1), values,
                  label=label)
    if log:
        axis.set_yscale("log")
    axis.legend(loc="upper right", ncols=1 if len(found) < 6 else 2)


def loss_curve(predictions, history, models, record, series=None, log=False, x="turn", rates=False, name=None,
               figures=None):
    figures = figures or Figure()
    if x not in ("turn", "step"):
        raise ValueError(f"loss_curve.x must be turn or step, got {x!r}")
    lines = History.read_steps(record) if x == "step" else History(history)
    found = lines.series(series)
    if not found:
        return None
    learning = lines.rates() if rates else {}
    positions = lines.positions(x)
    word = turn_word(record) if x == "turn" else "step"
    if learning:
        drawing, (axis, below) = figures.pyplot().subplots(2, 1, figsize=(8.0, 6.6), sharex=True,
                                                            gridspec_kw={"height_ratios": [3.0, 1.2]})
    else:
        drawing, axis = figures.single(width=8.0, height=5.0)
    draw_series(figures, axis, found, positions, log)
    last = ", ".join(f"{label} {values[-1]:.4g}" for label, values in list(found.items())[:4])
    figures.label(axis, "Training history" if x == "turn" else "Training steps", None if learning else word, "value",
                  note=f"last {word}: {last}" if last else None)
    if learning:
        values = [value for series_values in learning.values() for value in series_values if value > 0]
        draw_series(figures, below, learning, positions, bool(values) and max(values) / min(values) > 50)
        figures.label(below, None, word, "learning rate")
    figures.save(drawing, record, name or "loss_curve")
    return None
