import numpy

from .tables import finite, label_key, numbers_of, numeric, plain, rows_in, texts


QUANTILES = {"p05": 0.05, "p25": 0.25, "p50": 0.5, "p75": 0.75, "p95": 0.95}
COLORS = 20
LABELS = 50


def histogram(values, bins=40, edges=None):
    array = finite(values)
    if edges is not None:
        counts, found = numpy.histogram(array, bins=numpy.asarray(edges, dtype=numpy.float64))
    elif len(array):
        counts, found = numpy.histogram(array, bins=bins)
    else:
        counts, found = numpy.histogram(array, bins=bins, range=(0.0, 1.0))
    return {"edges": [float(edge) for edge in found], "counts": [int(count) for count in counts]}


def quantiles_of(values):
    if not len(values):
        return {key: None for key in QUANTILES}
    return {key: float(numpy.quantile(values, level)) for key, level in QUANTILES.items()}


def regression_stats(frame, pred, target, bins=40, sample=2000, worst=20, seed=0, residual_bins=None):
    guess = numbers_of(frame[pred])
    truth = numbers_of(frame[target])
    keep = numpy.isfinite(guess) & numpy.isfinite(truth)
    guess, truth, rows = guess[keep], truth[keep], rows_in(frame)[keep]
    count = int(len(guess))
    residual = guess - truth
    stats = {"count": count, "rmse": None, "mae": None, "r2": None, "bias": None,
             "residual_quantiles": {key: None for key in QUANTILES},
             "pred_hist": histogram(guess, bins), "truth_hist": histogram(truth, bins),
             "residual_hist": histogram(residual, bins if residual_bins is None else residual_bins),
             "scatter": {"row": [], "truth": [], "pred": []}, "worst": []}
    if count == 0:
        return stats
    edges = numpy.histogram_bin_edges(numpy.concatenate([guess, truth]), bins=bins)
    total = float(((truth - truth.mean()) ** 2).sum())
    stats["rmse"] = float(numpy.sqrt((residual ** 2).mean()))
    stats["mae"] = float(numpy.abs(residual).mean())
    stats["r2"] = None if total == 0 else float(1.0 - float((residual ** 2).sum()) / total)
    stats["bias"] = float(residual.mean())
    stats["residual_quantiles"] = quantiles_of(residual)
    stats["pred_hist"] = histogram(guess, edges=edges)
    stats["truth_hist"] = histogram(truth, edges=edges)
    chosen = numpy.arange(count)
    if sample is not None and count > int(sample):
        chosen = numpy.random.default_rng(seed).choice(count, int(sample), replace=False)
    chosen = chosen[numpy.argsort(rows[chosen], kind="stable")]
    stats["scatter"] = {"row": plain(rows[chosen]), "truth": plain(truth[chosen]), "pred": plain(guess[chosen])}
    order = numpy.argsort(-numpy.abs(residual), kind="stable")[:max(int(worst or 0), 0)]
    stats["worst"] = [{"row": row, "truth": float(truth[index]), "pred": float(guess[index]),
                       "error": float(residual[index])} for index, row in zip(order, plain(rows[order]))]
    return stats


def matrix_of(frame, columns):
    matrix = numpy.column_stack([numbers_of(frame[column]) for column in columns])
    return matrix, numpy.isfinite(matrix).all(axis=1)


def components(matrix):
    centered = matrix - matrix.mean(axis=0)
    _, spread, axes = numpy.linalg.svd(centered, full_matrices=False)
    variance = spread ** 2
    total = float(variance.sum())
    axes = axes[:2]
    signs = numpy.where(axes[numpy.arange(len(axes)), numpy.abs(axes).argmax(axis=1)] < 0, -1.0, 1.0)
    coordinates = centered @ (axes * signs[:, None]).T
    if coordinates.shape[1] < 2:
        coordinates = numpy.column_stack([coordinates, numpy.zeros((len(coordinates), 2 - coordinates.shape[1]))])
    explained = [float(variance[index] / total) if total > 0 and index < len(variance) else 0.0
                 for index in range(2)]
    return coordinates, explained


def color_column(frame, targets, images=None):
    columns = [str(column) for column in frame.columns]
    if "label" in columns:
        return "label"
    for target in targets:
        if target in columns:
            return target
    for column in columns:
        if column in ("row", images) or column.startswith(("pred_", "raw_", "prob_")):
            continue
        series = frame[column]
        if not numeric(series) and series.nunique(dropna=True) < COLORS:
            return column
    return None


def sampled(count, sample, seed=0):
    chosen = numpy.arange(count)
    if sample is not None and count > int(sample):
        chosen = numpy.sort(numpy.random.default_rng(seed).choice(count, int(sample), replace=False))
    return chosen


def embedding_stats(frame, columns, color=None, bins=40, sample=2000, seed=0):
    matrix, keep = matrix_of(frame, columns)
    matrix = matrix[keep]
    count = int(len(matrix))
    pca = {"x": [], "y": [], "explained": [None, None], "row": [], "index": [], "color": None, "color_by": color}
    stats = {"count": count, "dims": len(columns), "pca": pca, "norm_hist": histogram([], bins)}
    if not count:
        return stats
    coordinates, explained = components(matrix)
    chosen = sampled(count, sample, seed)
    positions = numpy.flatnonzero(keep)[chosen]
    pca.update({"x": plain(coordinates[chosen, 0]), "y": plain(coordinates[chosen, 1]), "explained": explained,
                "row": plain(rows_in(frame)[positions]), "index": [int(value) for value in frame.index[positions]]})
    if color is not None:
        pca["color"] = texts(frame[color].iloc[positions]).tolist()
    stats["norm_hist"] = histogram(numpy.linalg.norm(matrix, axis=1), bins)
    return stats


def label_counts(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    found = sorted(counts.items(), key=lambda item: (-item[1], label_key(item[0])))[:LABELS]
    return {"labels": [label for label, _ in found], "counts": [count for _, count in found]}


def values_stats(frame, pred, bins=40):
    series = frame[pred]
    if not numeric(series):
        found = [value for value in texts(series).tolist() if value is not None]
        return {"count": len(found), "mean": None, "std": None, "hist": None, "quantiles": None,
                "labels": label_counts(found)}
    values = finite(numbers_of(series))
    count = int(len(values))
    spread = {"min": float(values.min()) if count else None, **quantiles_of(values),
              "max": float(values.max()) if count else None}
    return {"count": count, "mean": float(values.mean()) if count else None,
            "std": float(values.std()) if count else None, "hist": histogram(values, bins), "quantiles": spread,
            "labels": None}
