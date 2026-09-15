from kalfa.std.common.figure import Figure
from kalfa.std.common.optional import load
from kalfa.std.plot.base import columns_of, first_set, set_frame


def sampled(table, sample, seed=0):
    if sample and len(table) > int(sample):
        return table.sample(int(sample), random_state=int(seed))
    return table


def kde(predictions, history, models, record, loaders=None, prep=None, sets=None, x=None, y=None, hue=None,
        sample=20000, fill=True, name=None, figures=None):
    figures = figures or Figure()
    seaborn = load("seaborn", "kde")
    set_name = first_set(sets, "train")
    table = set_frame(loaders, prep, set_name)
    if seaborn is None or table is None or x is None or x not in table.columns:
        return None
    data = sampled(table, sample)
    drawing, axis = figures.single(width=6.4, height=4.6)
    seaborn.kdeplot(data=data, x=x, y=y if y in data.columns else None, hue=hue if hue in data.columns else None,
                    fill=bool(fill), ax=axis, color=figures.categorical[0],
                    palette=figures.categorical if hue in data.columns else None)
    figures.label(axis, f"{x} and {y}" if y in data.columns else f"{x} density", x,
                 y if y in data.columns else "density", note=f"{len(data):,} rows of the {set_name} set")
    figures.save(drawing, record, name or "kde")
    return None


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
