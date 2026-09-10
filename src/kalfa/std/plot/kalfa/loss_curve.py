from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.common.history import History


@lego("/plot/kalfa/loss_curve", partial=True, alias="loss_curve",
      description="Every history series over the turns, or the named ones")
def loss_curve(predictions, history, models, record, series=None, log=False, name=None, figures=None):
    figures = figures or Figure()
    found = History(history).series(series)
    if not found:
        return None
    drawing, axis = figures.single(width=8.0, height=5.0)
    for label, values in found.items():
        axis.plot(range(1, len(values) + 1), values, label=label)
    if log:
        axis.set_yscale("log")
    axis.legend(loc="upper right", ncols=1 if len(found) < 6 else 2)
    last = ", ".join(f"{label} {values[-1]:.4g}" for label, values in list(found.items())[:4])
    figures.label(axis, "Training history", "turn", "value", note=f"last turn: {last}" if last else None)
    figures.save(drawing, record, name or "loss_curve")
    return None
