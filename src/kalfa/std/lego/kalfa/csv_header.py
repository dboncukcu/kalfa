import pandas

from kalfa.registration import lego


@lego("/lego/kalfa/csv_header",
      description="The columns, the dtypes of the first rows and the line count of a CSV file")
def csv_header(path, chunk=None):
    head = pandas.read_csv(path, nrows=64)
    with open(path, "rb") as stream:
        rows = max(sum(1 for _ in stream) - 1, 0)
    return {"columns": list(head.columns), "dtypes": {name: str(dtype) for name, dtype in head.dtypes.items()},
            "rows": rows}
