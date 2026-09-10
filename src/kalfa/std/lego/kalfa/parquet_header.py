from kalfa.registration import lego


@lego("/lego/kalfa/parquet_header",
      description="The columns, their arrow types and the row count of a parquet file, from its metadata")
def parquet_header(path, chunk=None):
    import pyarrow.parquet

    handle = pyarrow.parquet.ParquetFile(path)
    schema = handle.schema_arrow
    return {"columns": list(schema.names), "dtypes": {name: str(schema.field(name).type) for name in schema.names},
            "rows": handle.metadata.num_rows}
