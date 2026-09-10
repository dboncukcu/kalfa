from pathlib import Path

import pandas

from kalfa.registration import lego
from kalfa.std.common.files import read_json
from kalfa.std.source.kalfa.samples import ImageFolder, TextLines


def listed(names, columns, path):
    if columns is None:
        return list(names)
    missing = [name for name in columns if name not in names]
    if missing:
        raise KeyError(f"source columns {missing!r} are not in {path!r}; write names the file has")
    return list(columns)


@lego("/lego/kalfa/parquet_header",
      description="The columns, their arrow types and the row count of a parquet file, from its metadata; the "
                  "listed columns only when the source names them")
def parquet_header(path, chunk=None, columns=None):
    import pyarrow.parquet

    handle = pyarrow.parquet.ParquetFile(path)
    schema = handle.schema_arrow
    names = listed(schema.names, columns, path)
    return {"columns": names, "dtypes": {name: str(schema.field(name).type) for name in names},
            "rows": handle.metadata.num_rows}


@lego("/lego/kalfa/csv_header",
      description="The columns, the dtypes of the first rows and the line count of a CSV file; the listed columns "
                  "only when the source names them")
def csv_header(path, chunk=None, columns=None):
    head = pandas.read_csv(path, nrows=64)
    names = listed(list(head.columns), columns, path)
    with open(path, "rb") as stream:
        rows = max(sum(1 for _ in stream) - 1, 0)
    return {"columns": names, "dtypes": {name: str(head.dtypes[name]) for name in names}, "rows": rows}


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
