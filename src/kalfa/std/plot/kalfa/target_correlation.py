import numpy

from kalfa.registration import lego
from kalfa.std.common import figure
from kalfa.std.plot.base import bars, columns_of, first_set, set_frame


@lego("/plot/kalfa/target_correlation", partial=True, alias="target_correlation", refs={"target": "field"},
      description="The rank correlation of every column with the target, the strongest first; groups maps a "
                  "column to a group name and colours the bars by it")
def target_correlation(predictions, history, models, record, loaders=None, prep=None, sets=None, target=None,
                       method="spearman", columns=None, top=25, groups=None, name=None):
    set_name = first_set(sets, "train")
    table = set_frame(loaders, prep, set_name)
    if table is None:
        return None
    field = target or next(iter(prep.targets), None)
    if field is None or field not in table.columns:
        return None
    picked = columns_of(table, columns, skip=(field,) + tuple(prep.targets))
    if not picked:
        return None
    truth = table[field]
    found = [(column, float(table[column].corr(truth, method=method))) for column in picked]
    found = [(column, value) for column, value in found if numpy.isfinite(value)]
    found.sort(key=lambda item: abs(item[1]), reverse=True)
    found = found[:int(top)][::-1]
    if not found:
        return None
    names = [column for column, _ in found]
    drawing, axes = figure.sized(figure.width_of(9.5), 0.34 * len(names) + 2.0)
    axis = axes[0][0]
    bars(axis, names, [value for _, value in found], groups)
    figure.label(axis, f"Rank correlation with {field}", f"{method} rho", None,
                 note=f"{len(table):,} rows of the {set_name} set, the {len(names)} strongest")
    figure.save(drawing, record, name or "target_correlation")
    return None
