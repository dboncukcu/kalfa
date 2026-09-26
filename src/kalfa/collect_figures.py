import math
import re
import statistics


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def series(history, key):
    return [(line["turn"], float(line[key])) for line in history
            if finite(line.get(key)) and finite(line.get("turn"))]


def file_name(text):
    return re.sub(r"[^\w.-]+", "_", text).strip("_") or "value"


def curves(figures, target, entries, highlighted, monitor, unit, name="curves"):
    drawn = [(entry, series(entry["history"], monitor)) for entry in entries]
    if not any(points for _, points in drawn):
        return None
    drawing, axis = figures.single(width=8.0, height=4.6)
    shown = {id(entry) for entry in highlighted}
    for entry, points in drawn:
        if points and id(entry) not in shown:
            axis.plot(*zip(*points), color=figures.ink_muted, linewidth=0.8, alpha=0.35)
    labelled = False
    for position, entry in enumerate(highlighted):
        points = series(entry["history"], monitor)
        if points:
            axis.plot(*zip(*points), color=figures.categorical[position % len(figures.categorical)], linewidth=1.8,
                      label=entry["label"])
            labelled = True
    if labelled:
        axis.legend(loc="best")
    note = (f"{len(entries)} runs, {len(highlighted)} in colour" if len(entries) > len(highlighted)
            else f"{len(entries)} runs")
    figures.label(axis, f"{monitor} over the {unit}s", unit, monitor, note=note)
    return figures.save(drawing, target, name)


def spread(count):
    return [0.0] if count < 2 else [(position / (count - 1) - 0.5) * 0.3 for position in range(count)]


def against(figures, target, key, values, scores, monitor, log=False, levels=None, noun="points"):
    kept = [(value, score) for value, score in zip(values, scores) if finite(score) and value is not None]
    if not kept:
        return None
    drawing, axis = figures.single(width=6.0, height=4.0)
    if levels is not None:
        for position, level in enumerate(levels):
            here = [score for value, score in kept if value == level]
            if not here:
                continue
            axis.scatter([position + offset for offset in spread(len(here))], here, s=18,
                         color=figures.categorical[0], alpha=0.8)
            middle = statistics.median(here)
            axis.plot([position - 0.25, position + 0.25], [middle, middle], color=figures.ink, linewidth=1.6)
        axis.set_xticks(range(len(levels)), [str(level) for level in levels])
        axis.set_xlim(-0.6, len(levels) - 0.4)
    else:
        numeric = [(float(value), score) for value, score in kept if finite(value)]
        if not numeric:
            return None
        axis.scatter([value for value, _ in numeric], [score for _, score in numeric], s=18,
                     color=figures.categorical[0], alpha=0.8)
        if log and all(value > 0 for value, _ in numeric):
            axis.set_xscale("log")
    figures.label(axis, f"{monitor} against {key}", key, monitor,
                  note=f"{len(kept)} scored {noun}" + (", the line is the median of a level" if levels else ""))
    return figures.save(drawing, target, f"param_{file_name(key)}")
