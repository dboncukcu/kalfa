from pathlib import Path

import pandas

from kalfa.registration import lego
from kalfa.std.common.files import read_json
from kalfa.std.source.kalfa.samples import ImageFolder, TextLines


@lego("/lego/kalfa/parquet_header",
      description="The columns, their arrow types and the row count of a parquet file, from its metadata")
def parquet_header(path, chunk=None):
    import pyarrow.parquet

    handle = pyarrow.parquet.ParquetFile(path)
    schema = handle.schema_arrow
    return {"columns": list(schema.names), "dtypes": {name: str(schema.field(name).type) for name in schema.names},
            "rows": handle.metadata.num_rows}


@lego("/lego/kalfa/csv_header",
      description="The columns, the dtypes of the first rows and the line count of a CSV file")
def csv_header(path, chunk=None):
    head = pandas.read_csv(path, nrows=64)
    with open(path, "rb") as stream:
        rows = max(sum(1 for _ in stream) - 1, 0)
    return {"columns": list(head.columns), "dtypes": {name: str(dtype) for name, dtype in head.dtypes.items()},
            "rows": rows}


@lego("/lego/kalfa/image_folder_header",
      description="The fields, the dtypes, the image count and the classes of an image folder")
def image_folder_header(path):
    folder = ImageFolder(path)
    return {"columns": list(folder.fields), "dtypes": dict(folder.dtypes), "rows": len(folder),
            "classes": list(folder.classes)}


@lego("/lego/kalfa/text_lines_header", description="The text field and the line count of a text file")
def text_lines_header(path):
    return {"columns": ["text"], "dtypes": {"text": "string"}, "rows": len(TextLines(path))}


@lego("/lego/kalfa/prepared_header",
      description="The header a prepared directory recorded in its manifest: the columns, the dtypes and the rows")
def prepared_header(path):
    manifest = read_json(Path(path) / "manifest.json")
    if manifest is None or manifest.get("kind") != "data":
        raise ValueError(f"{path} is no prepared directory; kalfa prepare writes one")
    return dict(manifest["header"])
