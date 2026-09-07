"""Sources: legos that read the data into a DataFrame."""

from pathlib import Path


from ..registration import lego
from .samples import Samples

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


@lego("/source/kalfa/parquet", returns="df", alias="parquet",
            description="Read a parquet file into a DataFrame")
def parquet(path):
    import pandas

    return pandas.read_parquet(path)


@lego("/source/kalfa/csv", returns="df", alias="csv",
            description="Read a CSV file into a DataFrame")
def csv(path):
    import pandas

    return pandas.read_csv(path)


@lego("/source/kalfa/parquet_stream", returns="df",
            description="Read a parquet file in chunks (the lazy set): a stream the data legos filter, cut and fit "
                        "without loading the table")
def parquet_stream(path, chunk=65536):
    from .stream import ParquetChunks, Stream

    return Stream(ParquetChunks(path, chunk))


@lego("/source/kalfa/csv_stream", returns="df",
            description="Read a CSV file in chunks (the lazy set)")
def csv_stream(path, chunk=65536):
    from .stream import CsvChunks, Stream

    return Stream(CsvChunks(path, chunk))


class ImageFolder:
    """root/<class>/<file>: one item per image with fields ``image`` (a PIL image, loaded on access) and ``label``
    (the class index in sorted class order)."""

    fields = ["image", "label"]
    dtypes = {"image": "image", "label": "int64"}

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"image folder {root!r} does not exist")
        self.classes = sorted(entry.name for entry in self.root.iterdir() if entry.is_dir())
        self.samples = []
        for position, name in enumerate(self.classes):
            for path in sorted((self.root / name).iterdir()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    self.samples.append((path, position))
        self.labels = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, position):
        from PIL import Image

        path, label = self.samples[position]
        with Image.open(path) as image:
            image.load()
            return {"image": image.copy(), "label": label}

    def column(self, name):
        if name == "label":
            return list(self.labels)
        raise ValueError(f"field {name!r} is not readable as a column; only label is")


class TextLines:
    """The non empty lines of a text file, one item per line with the field ``text``."""

    fields = ["text"]
    dtypes = {"text": "string"}

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"text file {path!r} does not exist")
        self.lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, position):
        return {"text": self.lines[position]}

    def column(self, name):
        if name == "text":
            return list(self.lines)
        raise ValueError(f"field {name!r} is not readable as a column; only text is")


@lego("/source/kalfa/text_lines", returns="df", alias="text_lines",
            description="The lines of a text file as a Dataset with the field text")
def text_lines(path):
    return Samples(TextLines(path))


@lego("/source/kalfa/image_folder", returns="df", alias="image_folder",
            description="Images under root/<class>/ as a Dataset with fields image and label")
def image_folder(path):
    return Samples(ImageFolder(path))


def header(uri, params):
    """Column names, dtypes and the row count of a source, read from the file header only."""
    path = params.get("path")
    if uri in ("/source/kalfa/parquet", "/source/kalfa/parquet_stream"):
        import pyarrow.parquet

        handle = pyarrow.parquet.ParquetFile(path)
        schema = handle.schema_arrow
        dtypes = {name: str(schema.field(name).type) for name in schema.names}
        return {"columns": list(schema.names), "dtypes": dtypes, "rows": handle.metadata.num_rows}
    if uri == "/source/kalfa/image_folder":
        folder = ImageFolder(path)
        return {"columns": list(folder.fields), "dtypes": dict(folder.dtypes), "rows": len(folder),
                "classes": list(folder.classes)}
    if uri == "/source/kalfa/text_lines":
        lines = TextLines(path)
        return {"columns": ["text"], "dtypes": {"text": "string"}, "rows": len(lines)}
    if uri in ("/source/kalfa/csv", "/source/kalfa/csv_stream"):
        import pandas

        head = pandas.read_csv(path, nrows=64)
        with open(path, "rb") as stream:
            rows = max(sum(1 for _ in stream) - 1, 0)
        return {"columns": list(head.columns), "dtypes": {name: str(dtype) for name, dtype in head.dtypes.items()},
                "rows": rows}
    return None


STREAM_SOURCES = ("/source/kalfa/parquet_stream", "/source/kalfa/csv_stream")
