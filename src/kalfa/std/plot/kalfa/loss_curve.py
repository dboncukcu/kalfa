from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.common.history import History


@lego("/plot/kalfa/loss_curve", partial=True, alias="loss_curve",
      description="Every history series over the turns, or the named ones; x: step draws the per update series "
                  "of steps.jsonl (the loss, the gradient norm of every optimizer) over the steps instead")
def loss_curve(predictions, history, models, record, series=None, log=False, x="turn", name=None, figures=None):
    figures = figures or Figure()
    if x not in ("turn", "step"):
        raise ValueError(f"loss_curve.x must be turn or step, got {x!r}")
    lines = History.read_steps(record) if x == "step" else History(history)
    found = lines.series(series)
    if not found:
        return None
    drawing, axis = figures.single(width=8.0, height=5.0)
    positions = lines.positions(x)
    for label, values in found.items():
        axis.plot(positions[:len(values)] if len(positions) >= len(values) else range(1, len(values) + 1), values,
                  label=label)
    if log:
        axis.set_yscale("log")
    axis.legend(loc="upper right", ncols=1 if len(found) < 6 else 2)
    last = ", ".join(f"{label} {values[-1]:.4g}" for label, values in list(found.items())[:4])
    figures.label(axis, "Training history" if x == "turn" else "Training steps", x, "value",
                  note=f"last {x}: {last}" if last else None)
    figures.save(drawing, record, name or "loss_curve")
    return None
