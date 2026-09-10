from kalfa.registration import lego
from kalfa.std.common import figure
from kalfa.std.plot.base import columns_of, first_set, set_frame


@lego("/plot/kalfa/correlation_heatmap", partial=True, alias="correlation_heatmap",
      description="The rank correlation of every column of a set against every other, features and targets "
                  "together; it reads the set the definition names (train without one)")
def correlation_heatmap(predictions, history, models, record, loaders=None, prep=None, sets=None,
                        method="spearman", columns=None, sample=80000, annotate=False, name=None):
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
    drawing, axes = figure.sized(side, side * 0.92)
    axis = axes[0][0]
    drawn = axis.imshow(matrix, cmap=figure.diverging(), vmin=-1, vmax=1)
    axis.set_xticks(range(len(picked)), picked, rotation=90, fontsize=8)
    axis.set_yticks(range(len(picked)), picked, fontsize=8)
    axis.grid(visible=False)
    if annotate and len(picked) <= 20:
        for row in range(len(picked)):
            for column in range(len(picked)):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=7,
                          color=figure.INK if abs(matrix[row, column]) < 0.6 else "#ffffff")
    figure.colorbar(drawing, drawn, axis, f"{method} rho", fraction=0.032)
    figure.label(axis, "Column correlation", None, None,
                 note=f"{len(data):,} rows of the {set_name} set, {len(picked)} columns")
    figure.save(drawing, record, name or "correlation_heatmap")
    return None
