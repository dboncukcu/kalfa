from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.plot.base import first_set, set_frame
from kalfa.std.common.optional import load
from kalfa.std.plot.seaborn.base import sampled


@lego("/plot/seaborn/violin", partial=True, alias="violin", requires="seaborn",
      refs={"value": "column", "group": "column"},
      description="seaborn's violin of one column of a set, split by a grouping column when one is named; "
                  "skipped with a warning when seaborn is not installed")
def violin(predictions, history, models, record, loaders=None, prep=None, sets=None, value=None, group=None,
           sample=20000, name=None, figures=None):
    figures = figures or Figure()
    seaborn = load("seaborn", "violin")
    set_name = first_set(sets, "train")
    table = set_frame(loaders, prep, set_name)
    if seaborn is None or table is None or value is None or value not in table.columns:
        return None
    data = sampled(table, sample)
    drawing, axis = figures.single(width=7.0, height=4.6)
    seaborn.violinplot(data=data, x=group if group in data.columns else None, y=value, ax=axis,
                       color=figures.categorical[0], palette=figures.categorical if group in data.columns else None,
                       hue=group if group in data.columns else None, legend=False)
    figures.label(axis, f"{value} by {group}" if group in data.columns else value, group, value,
                 note=f"{len(data):,} rows of the {set_name} set")
    figures.save(drawing, record, name or "violin")
    return None
