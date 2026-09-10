from kalfa.registration import lego
from kalfa.std.transform.base import table_only


@lego("/transform/kalfa/derive", alias="derive", partial=True, needs_table=True,
      description="A new column from a pandas eval expression over the frame (log10(x), a > b, a + b)")
def derive(df, column, expr):
    table = table_only(df, "derive")
    return table.assign(**{column: table.eval(expr)})
