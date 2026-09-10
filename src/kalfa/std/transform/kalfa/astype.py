from kalfa.registration import lego
from kalfa.std.transform.base import table_only


@lego("/transform/kalfa/astype", alias="astype", partial=True, needs_table=True,
      description="Cast columns to dtypes, {column: dtype}")
def astype(df, columns):
    return table_only(df, "astype").astype(dict(columns))
