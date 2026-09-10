import pandas


def table_only(df, what):
    if not isinstance(df, pandas.DataFrame):
        raise ValueError(f"{what} needs a table in memory; a stream or a Dataset source keeps its rows as they are and "
                         f"filter is the transform they take")
    return df
