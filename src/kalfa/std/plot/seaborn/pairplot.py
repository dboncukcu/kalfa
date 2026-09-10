from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.plot.base import columns_of, first_set, set_frame
from kalfa.std.common.optional import load
from kalfa.std.plot.seaborn.base import sampled


@lego("/plot/seaborn/pairplot", partial=True, needs=["train_loader"], alias="pairplot", requires="seaborn",
      description="seaborn's pairwise grid of a few columns of a set, hue colouring the points by a column; "
                  "skipped with a warning when seaborn is not installed")
def pairplot(predictions, history, models, record, loaders=None, prep=None, sets=None, columns=None, hue=None,
             sample=5000, kind="scatter", diagonal="hist", height=2.2, name=None, figures=None):
    figures = figures or Figure()
    seaborn = load("seaborn", "pairplot")
    set_name = first_set(sets, "train")
    table = set_frame(loaders, prep, set_name)
    if seaborn is None or table is None:
        return None
    picked = columns_of(table, columns)[:8]
    if len(picked) < 2:
        return None
    wanted = picked + ([hue] if hue and hue in table.columns and hue not in picked else [])
    data = sampled(table[wanted], sample)
    grid = seaborn.pairplot(data, vars=picked, hue=hue if hue in data.columns else None, kind=kind,
                            diag_kind=diagonal, height=float(height),
                            palette=figures.categorical if hue in data.columns else None,
                            plot_kws={"color": figures.categorical[0], "edgecolor": "none", "s": 12}
                            if hue not in data.columns else None)
    figures.title(grid.figure, f"Pairwise columns ({set_name} set, {len(data):,} rows)")
    figures.save(grid.figure, record, name or "pairplot")
    return None
