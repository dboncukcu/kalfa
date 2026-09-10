from kalfa.registration import lego
from kalfa.std.transform.base import table_only


@lego("/transform/kalfa/drop", alias="drop", partial=True, needs_table=True,
      description="Drop columns from the frame before anything reads them")
def drop(df, columns):
    return table_only(df, "drop").drop(columns=list(columns))
