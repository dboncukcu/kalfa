import re

import pandas


def table_only(df, what):
    if not isinstance(df, pandas.DataFrame):
        raise ValueError(f"{what} needs a table in memory; a stream or a Dataset source keeps its rows as they are and "
                         f"filter is the transform they take")
    return df


def filter_rows(df, query):
    return df.query(query)


def derive(df, column, expr):
    table = table_only(df, "derive")
    return table.assign(**{column: table.eval(expr)})


def rename(df, pattern, to):
    return table_only(df, "rename").rename(columns=lambda name: re.sub(pattern, to, name))


def astype(df, columns):
    return table_only(df, "astype").astype(dict(columns))


def drop(df, columns):
    return table_only(df, "drop").drop(columns=list(columns))
