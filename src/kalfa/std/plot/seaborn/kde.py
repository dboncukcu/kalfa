from kalfa.registration import lego
from kalfa.std.common import figure
from kalfa.std.plot.base import first_set, set_frame
from kalfa.std.plot.seaborn.base import sampled, seaborn_module


@lego("/plot/seaborn/kde", partial=True, alias="kde", refs={"x": "column", "y": "column", "hue": "column"},
      description="seaborn's kernel density of one column of a set, or of two as contours; skipped with a "
                  "warning when seaborn is not installed")
def kde(predictions, history, models, record, loaders=None, prep=None, sets=None, x=None, y=None, hue=None,
        sample=20000, fill=True, name=None):
    seaborn = seaborn_module("kde")
    set_name = first_set(sets, "train")
    table = set_frame(loaders, prep, set_name)
    if seaborn is None or table is None or x is None or x not in table.columns:
        return None
    data = sampled(table, sample)
    drawing, axis = figure.single(width=6.4, height=4.6)
    seaborn.kdeplot(data=data, x=x, y=y if y in data.columns else None, hue=hue if hue in data.columns else None,
                    fill=bool(fill), ax=axis, color=figure.CATEGORICAL[0],
                    palette=figure.CATEGORICAL if hue in data.columns else None)
    figure.label(axis, f"{x} and {y}" if y in data.columns else f"{x} density", x,
                 y if y in data.columns else "density", note=f"{len(data):,} rows of the {set_name} set")
    figure.save(drawing, record, name or "kde")
    return None
