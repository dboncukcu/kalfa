import numpy
import pandas

from kalfa.registration import lego
from kalfa.std.common.log import logger_for
from kalfa.std.common.samples import is_samples
from kalfa.std.common.stream import is_stream
from kalfa.std.pre.base import (SampleFrame, StreamFrame, StreamView, TableFrame, cast_values, run_chain, sets_of,
                                typed_extras, values_of)


logger = logger_for("data.prep")


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
    extra = df[[column for column in df.columns if column not in named and column not in prep.drop]]
    return TableFrame(prep.features, prep.targets, set, data=data, extra=extra, mask=kept_rows(df, mask))


@lego("/lego/kalfa/apply",
      description="Apply the fitted chains to one set and type its columns; keys carry the sets a "
                  "preprocessor is limited to; the rows the mask query selects stay in the frame and are not "
                  "scored, the loader leaves them out and the plots see them as masked")
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
