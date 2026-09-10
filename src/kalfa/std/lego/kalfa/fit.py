import copy

from kalfa.registration import lego
from kalfa.std.common.log import clock, logger_for
from kalfa.std.common.samples import is_samples
from kalfa.std.common.stream import is_stream
from kalfa.std.pre.base import (
    Prep,
    cast_values,
    columns_of,
    extra_columns,
    fit_chains,
    fit_stream,
    report_fitted,
    resolve_fields,
    sets_of,
    torch_dtype,
    values_of,
    write_prep,
)


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


@lego("/lego/kalfa/fit", returns="prep", state=True, bus=["record"],
      description="Resolve the field globs and fit every preprocessor chain on the train set; keys carry the "
                  "sets a preprocessor is limited to")
def fit(df, fields, preprocessors, drop, keys=None, record=None):
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
    prep = Prep(items, fitted, sets, dtypes, list(drop or []))
    if record is not None:
        write_prep(prep, record)
    return report_fitted(prep, started)
