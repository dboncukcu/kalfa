import numpy

from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.plot.base import bars, columns_of, first_set, logged, set_frame


@lego("/plot/kalfa/target_vs_features", partial=True, alias="target_vs_features", refs={"target": "field"},
      needs=["train_loader"],
      description="One panel per feature: the target against it as a hexbin density with the median profile "
                  "over equal count bins; it reads the set the definition names (train without one) and "
                  "draws in the original units")
def target_vs_features(predictions, history, models, record, loaders=None, prep=None, sets=None, target=None,
                       columns=None, log=None, gridsize=60, bins=60, limit=24, per_row=4, name=None, figures=None):
    figures = figures or Figure()
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
    drawing, axes = figures.grid(rows, width, width=6.4, height=4.2)
    panels = [axis for row in axes for axis in row]
    truth = table[field].to_numpy(dtype="float64")
    for axis, column in zip(panels, picked):
        values, shown = logged(table[column].to_numpy(dtype="float64"), column, log)
        x, y = figures.finite(values, truth)
        if not len(x):
            axis.axis("off")
            continue
        figures.density(drawing, axis, x, y, gridsize)
        centers, profile = figures.profile(x, y, bins)
        if len(centers):
            axis.plot(centers, profile, color=figures.categorical[1], linewidth=2.4, label="median profile")
            axis.legend(loc="lower left")
        figures.label(axis, f"{field} vs {shown}", shown, field)
    for axis in panels[len(picked):]:
        axis.axis("off")
    figures.title(drawing, f"{field} against every feature ({set_name} set, {len(table):,} rows)")
    drawing.tight_layout(rect=(0, 0, 1, 0.97))
    figures.save(drawing, record, name or "target_vs_features")
    return None


@lego("/plot/kalfa/target_correlation", partial=True, alias="target_correlation", refs={"target": "field"},
      needs=["train_loader"],
      description="The rank correlation of every column with the target, the strongest first; groups maps a "
                  "column to a group name and colours the bars by it")
def target_correlation(predictions, history, models, record, loaders=None, prep=None, sets=None, target=None,
                       method="spearman", columns=None, top=25, groups=None, name=None, figures=None):
    figures = figures or Figure()
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
    drawing, axes = figures.sized(figures.width_of(9.5), 0.34 * len(names) + 2.0)
    axis = axes[0][0]
    bars(figures, axis, names, [value for _, value in found], groups)
    figures.label(axis, f"Rank correlation with {field}", f"{method} rho", None,
                 note=f"{len(table):,} rows of the {set_name} set, the {len(names)} strongest")
    figures.save(drawing, record, name or "target_correlation")
    return None


@lego("/plot/kalfa/correlation_heatmap", partial=True, needs=["train_loader"], alias="correlation_heatmap",
      description="The rank correlation of every column of a set against every other, features and targets "
                  "together; it reads the set the definition names (train without one)")
def correlation_heatmap(predictions, history, models, record, loaders=None, prep=None, sets=None,
                        method="spearman", columns=None, sample=80000, annotate=False, name=None, figures=None):
    figures = figures or Figure()
    set_name = first_set(sets, "train")
    table = set_frame(loaders, prep, set_name)
    if table is None:
        return None
    picked = columns_of(table, columns)
    if len(picked) < 2:
        return None
    data = table[picked]
    if sample and len(data) > int(sample):
        data = data.sample(int(sample), random_state=0)
    matrix = data.corr(method=method).to_numpy()
    side = 0.34 * len(picked) + 3.4
    drawing, axes = figures.sized(side, side * 0.92)
    axis = axes[0][0]
    drawn = axis.imshow(matrix, cmap=figures.diverging(), vmin=-1, vmax=1)
    axis.set_xticks(range(len(picked)), picked, rotation=90, fontsize=8)
    axis.set_yticks(range(len(picked)), picked, fontsize=8)
    axis.grid(visible=False)
    if annotate and len(picked) <= 20:
        for row in range(len(picked)):
            for column in range(len(picked)):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=7,
                          color=figures.ink if abs(matrix[row, column]) < 0.6 else "#ffffff")
    figures.colorbar(drawing, drawn, axis, f"{method} rho", fraction=0.032)
    figures.label(axis, "Column correlation", None, None,
                 note=f"{len(data):,} rows of the {set_name} set, {len(picked)} columns")
    figures.save(drawing, record, name or "correlation_heatmap")
    return None


@lego("/plot/kalfa/feature_distributions", partial=True, needs=["train_loader"], alias="feature_distributions",
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
