from kalfa.registration import lego
from kalfa.std.common import figure
from kalfa.std.plot.base import columns_of, first_set, logged, set_frame


@lego("/plot/kalfa/target_vs_features", partial=True, alias="target_vs_features", refs={"target": "field"},
      description="One panel per feature: the target against it as a hexbin density with the median profile "
                  "over equal count bins; it reads the set the definition names (train without one) and "
                  "draws in the original units")
def target_vs_features(predictions, history, models, record, loaders=None, prep=None, sets=None, target=None,
                       columns=None, log=None, gridsize=60, bins=60, limit=24, per_row=4, name=None):
    set_name = first_set(sets, "train")
    table = set_frame(loaders, prep, set_name)
    if table is None:
        return None
    field = target or next(iter(prep.targets), None)
    if field is None or field not in table.columns:
        return None
    picked = columns_of(table, columns, skip=(field,) + tuple(prep.targets))[:int(limit)]
    if not picked:
        return None
    width = max(1, min(int(per_row or 4), len(picked)))
    rows = -(-len(picked) // width)
    drawing, axes = figure.grid(rows, width, width=6.4, height=4.2)
    panels = [axis for row in axes for axis in row]
    truth = table[field].to_numpy(dtype="float64")
    for axis, column in zip(panels, picked):
        values, shown = logged(table[column].to_numpy(dtype="float64"), column, log)
        x, y = figure.finite(values, truth)
        if not len(x):
            axis.axis("off")
            continue
        figure.density(drawing, axis, x, y, gridsize)
        centers, profile = figure.profile(x, y, bins)
        if len(centers):
            axis.plot(centers, profile, color=figure.CATEGORICAL[1], linewidth=2.4, label="median profile")
            axis.legend(loc="lower left")
        figure.label(axis, f"{field} vs {shown}", shown, field)
    for axis in panels[len(picked):]:
        axis.axis("off")
    figure.title(drawing, f"{field} against every feature ({set_name} set, {len(table):,} rows)")
    drawing.tight_layout(rect=(0, 0, 1, 0.97))
    figure.save(drawing, record, name or "target_vs_features")
    return None
