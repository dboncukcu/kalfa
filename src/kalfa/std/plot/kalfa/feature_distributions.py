from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.plot.base import columns_of, first_set, logged, set_frame


@lego("/plot/kalfa/feature_distributions", partial=True, alias="feature_distributions",
      description="A histogram per feature column of a set, in the original units; log names the columns to "
                  "draw on a log10 axis")
def feature_distributions(predictions, history, models, record, loaders=None, prep=None, sets=None, columns=None,
                          log=None, bins=80, limit=24, per_row=4, name=None, figures=None):
    figures = figures or Figure()
    set_name = first_set(sets, "train")
    table = set_frame(loaders, prep, set_name)
    if table is None:
        return None
    picked = columns_of(table, columns)[:int(limit)]
    if not picked:
        return None
    width = max(1, min(int(per_row or 4), len(picked)))
    rows = -(-len(picked) // width)
    drawing, axes = figures.grid(rows, width, width=3.4, height=2.6)
    panels = [axis for row in axes for axis in row]
    for axis, column in zip(panels, picked):
        values, shown = logged(table[column].to_numpy(dtype="float64"), column, log)
        values = figures.finite(values)[0]
        if not len(values):
            axis.axis("off")
            continue
        axis.hist(values, bins=int(bins), color=figures.categorical[0], edgecolor="none")
        axis.tick_params(labelsize=8)
        figures.label(axis, shown, None, None)
    for axis in panels[len(picked):]:
        axis.axis("off")
    figures.title(drawing, f"Column distributions ({set_name} set, {len(table):,} rows)")
    drawing.tight_layout(rect=(0, 0, 1, 0.97))
    figures.save(drawing, record, name or "feature_distributions")
    return None
