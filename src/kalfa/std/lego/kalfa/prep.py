import copy

import numpy
import pandas

from kalfa.std.common.log import clock, logger_for
from kalfa.std.common.samples import is_samples
from kalfa.std.common.stream import is_stream
from kalfa.std.pre.base import (Prep, SampleFrame, StreamFrame, StreamView, TableFrame, cast_values, columns_of,
                                extra_columns, fit_chains, fit_stream, matches_any, read_prep, report_fitted,
                                resolve_fields, run_chain, sets_of, torch_dtype, typed_extras, values_of, write_prep)


logger = logger_for("data.prep")


def fit_samples(items, df, templates, dtypes):
    fitted = {}
    for item in items:
        kind = df.dtypes.get(item.name)
        for name in item.chain:
            preprocessor = copy.deepcopy(templates[name])
            if preprocessor.fits:
                try:
                    values = df.column(item.name)
                except ValueError:
                    raise ValueError(f"preprocessor {name!r} fits on a column, but field {item.name!r} of the "
                                     f"Dataset source cannot be read as one") from None
                preprocessor.fit(values)
            fitted.setdefault(name, {})[item.name] = preprocessor
            kind = preprocessor.dtype or kind
        item.columns = [item.name]
        resolved = torch_dtype(kind)
        if resolved is None:
            raise TypeError(f"field {item.name!r} has dtype {kind} after its chain; add a preprocessor that turns "
                            f"it into a tensor (to_tensor)")
        dtypes[item.name] = resolved
    return fitted


def fit_table(items, df, templates, sets, dtypes):
    fitted = {}
    final, sides = fit_chains(items, lambda item: values_of(df, item.name), templates, sets, fitted)
    for item in items:
        values = final[item.name]
        if len(values):
            values, kind = cast_values(values, item.name)
        else:
            kind = torch_dtype(values.dtype) or "float32"
        item.columns = columns_of(item, values, fitted)
        for column in item.columns:
            dtypes[column] = kind
        item.extras = extra_columns(sides.get(item.name, {}), dtypes)
    return fitted


def sample_frame(df, prep, set, sets):
    chains = {}
    for item in prep.fields:
        if item.name not in df.fields:
            raise ValueError(f"the {set} data lacks field {item.name!r}")
        chains[item.name] = [prep.object_of(name, item.name) for name in item.chain
                             if sets.get(name) is None or set in sets[name]]
    return SampleFrame([], prep.targets, set, dataset=df, fields=[item.name for item in prep.fields], chains=chains)


def kept_rows(df, mask):
    if mask is None:
        return None
    return ~numpy.asarray(df.eval(mask), dtype=bool)


def table_frame(df, prep, set, sets, mask=None):
    columns = {}
    for item in prep.fields:
        if item.name not in df.columns:
            raise ValueError(f"the {set} data lacks column {item.name!r}")
        values, extras = run_chain(item, values_of(df, item.name), prep.fitted, {}, sets, set, fit=False)
        if len(values):
            values, _ = cast_values(values, item.name)
        if values.ndim == 1:
            columns[item.columns[0]] = values
        else:
            for position, column in enumerate(item.columns):
                columns[column] = values[:, position]
        columns.update(typed_extras(extras, prep.dtypes))
    data = pandas.DataFrame(columns, index=df.index)
    if len(data) == 0:
        data = pandas.DataFrame({column: numpy.zeros(0, dtype=prep.dtypes[column]) for column in prep.dtypes},
                                index=df.index)
    named = {item.name for item in prep.fields}
    extra = df[[column for column in df.columns
                if column not in named and matches_any(column, prep.spectators)]]
    return TableFrame(prep.features, prep.targets, set, data=data, extra=extra, mask=kept_rows(df, mask))


def fit(df, fields, preprocessors, drop, keys=None, record=None, spectators=None):
    started = clock()
    templates = dict(preprocessors or {})
    unknown = sorted({name for spec in (fields or {}).values() for name in ((spec or {}).get("preprocessors") or [])
                      if name not in templates})
    if unknown:
        raise ValueError(f"fields name preprocessors {unknown} that data.preprocessors does not define")
    items = resolve_fields(df, fields or {}, drop)
    if templates:
        logger.info(f"fitting {len(templates)} preprocessors over {len(items)} fields")
    sets = {name: list(allowed) for name, allowed in sets_of(keys).items()}
    dtypes = {}
    if is_stream(df):
        fitted = {}
        fit_stream(items, df, templates, sets, fitted, dtypes)
    elif is_samples(df):
        fitted = fit_samples(items, df, templates, dtypes)
    else:
        fitted = fit_table(items, df, templates, sets, dtypes)
    prep = Prep(items, fitted, sets, dtypes, list(drop or []), list(spectators or []))
    if record is not None:
        write_prep(prep, record)
    return report_fitted(prep, started)


def prep_of(record):
    return read_prep(record)


def apply(df, prep, set, keys=None, mask=None):
    logger.debug(f"applying the chains to the {set} set")
    sets = sets_of(keys) if keys is not None else prep.sets
    if is_stream(df):
        missing = [item.name for item in prep.fields if item.name not in df.columns]
        if missing:
            raise ValueError(f"the {set} data lacks columns {missing}")
        return StreamFrame(prep.features, prep.targets, set, stream=StreamView(df, prep, set, sets))
    if is_samples(df):
        return sample_frame(df, prep, set, sets)
    return table_frame(df, prep, set, sets, mask)
