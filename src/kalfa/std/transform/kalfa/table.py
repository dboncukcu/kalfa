import re

import pandas

from kalfa.registration import lego


def table_only(df, what):
    if not isinstance(df, pandas.DataFrame):
        raise ValueError(f"{what} needs a table in memory; a stream or a Dataset source keeps its rows as they are and "
                         f"filter is the transform they take")
    return df


@lego("/transform/kalfa/filter", alias="filter", partial=True,
      description="Keep the rows a pandas query selects; a bare string in data.transform is this call; a stream "
                  "applies it chunk by chunk and a Dataset source takes field equality queries")
def filter_rows(df, query):
    return df.query(query)


@lego("/transform/kalfa/derive", alias="derive", partial=True, needs_table=True,
      description="A new column from a pandas eval expression over the frame (log10(x), a > b, a + b)")
def derive(df, column, expr):
    table = table_only(df, "derive")
    return table.assign(**{column: table.eval(expr)})


@lego("/transform/kalfa/rename", alias="rename", partial=True, needs_table=True,
      description="Rename the columns a regular expression matches, pattern to the replacement, backreferences "
                  "allowed (cms_(.*)_Z_score to z_\\1)")
def rename(df, pattern, to):
    return table_only(df, "rename").rename(columns=lambda name: re.sub(pattern, to, name))


@lego("/transform/kalfa/astype", alias="astype", partial=True, needs_table=True,
      description="Cast columns to dtypes, {column: dtype}")
def astype(df, columns):
    return table_only(df, "astype").astype(dict(columns))


@lego("/transform/kalfa/drop", alias="drop", partial=True, needs_table=True,
      description="Drop columns from the frame before anything reads them")
def drop(df, columns):
    return table_only(df, "drop").drop(columns=list(columns))
