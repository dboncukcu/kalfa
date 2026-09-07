"""Stream sources: a table read in chunks, with a row window (the sequential split) and filters applied on the way.

The lazy set (CONFIG.md section 3): no random split, kfold or balanced sampler; filters run on the stream; the train pass
shuffles through a buffer; a train stream that yields nothing is an error at the first pass.
"""

from pathlib import Path

import numpy


class ParquetChunks:
    """Row groups of a parquet file in chunks of ``chunk`` rows; the row count comes from the metadata."""

    def __init__(self, path, chunk=65536):
        import pyarrow.parquet

        self.path = str(path)
        self.chunk = int(chunk)
        if not Path(self.path).is_file():
            raise FileNotFoundError(f"parquet file {path!r} does not exist")
        handle = pyarrow.parquet.ParquetFile(self.path)
        self.columns = list(handle.schema_arrow.names)
        self.rows = int(handle.metadata.num_rows)

    def chunks(self):
        import pyarrow.parquet

        handle = pyarrow.parquet.ParquetFile(self.path)
        offset = 0
        for batch in handle.iter_batches(batch_size=self.chunk):
            frame = batch.to_pandas()
            frame.index = range(offset, offset + len(frame))
            offset += len(frame)
            yield frame


class CsvChunks:
    """A CSV file in chunks of ``chunk`` rows; the row count is one line count."""

    def __init__(self, path, chunk=65536):
        import pandas

        self.path = str(path)
        self.chunk = int(chunk)
        if not Path(self.path).is_file():
            raise FileNotFoundError(f"csv file {path!r} does not exist")
        self.columns = list(pandas.read_csv(self.path, nrows=0).columns)
        with open(self.path, "rb") as stream:
            self.rows = max(sum(1 for _ in stream) - 1, 0)

    def chunks(self):
        import pandas

        offset = 0
        for frame in pandas.read_csv(self.path, chunksize=self.chunk):
            frame.index = range(offset, offset + len(frame))
            offset += len(frame)
            yield frame


class Stream:
    """A chunked table as the data legos see it: a reader, a row window and the queries applied per chunk."""

    def __init__(self, reader, start=0, stop=None, queries=()):
        self.reader = reader
        self.start = int(start)
        self.stop = reader.rows if stop is None else int(stop)
        self.queries = tuple(queries)

    @property
    def columns(self):
        return list(self.reader.columns)

    @property
    def rows(self):
        """The rows of the window before the filters."""
        return max(self.stop - self.start, 0)

    def window(self, start, stop):
        return Stream(self.reader, self.start + start, min(self.start + stop, self.stop), self.queries)

    def query(self, text):
        return Stream(self.reader, self.start, self.stop, self.queries + (text,))

    def empty(self):
        return Stream(self.reader, self.start, self.start, self.queries)

    def chunks(self):
        for frame in self.reader.chunks():
            first, last = int(frame.index[0]), int(frame.index[-1]) + 1
            if last <= self.start:
                continue
            if first >= self.stop:
                break
            if first < self.start or last > self.stop:
                frame = frame.loc[max(first, self.start):min(last, self.stop) - 1]
            for text in self.queries:
                frame = frame.query(text)
            if len(frame):
                yield frame

    def head(self):
        """The first chunk after the window and the filters, or None."""
        return next(iter(self.chunks()), None)

    def count(self):
        return sum(len(frame) for frame in self.chunks())


def is_stream(value):
    return isinstance(value, Stream)


def like(stream, path):
    """A stream over another file read the way this one is (same reader class and chunk size)."""
    return Stream(type(stream.reader)(path, stream.reader.chunk))


def positions(stream):
    return numpy.concatenate([numpy.asarray(frame.index) for frame in stream.chunks()] or [numpy.zeros(0, dtype="int64")])
