from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pandas
import torch
from torch.utils.data import IterableDataset, get_worker_info

from kalfa.core.errors import BuildError
from kalfa.std.common import positive_int
from kalfa.std.data import splits
from kalfa.std.data.base import config_directories
from kalfa.std.data.options import STREAM_SPLIT, locate, query


DEFAULTS = {"chunk": 65536, "fit_rows": 100000}
PARQUET = (".parquet", ".pq")
SEPARATORS = {".csv": ",", ".tsv": "\t"}
INDEX_PREFIX = "__index_level_"


def joined(pieces):
    return pieces[0] if len(pieces) == 1 else pandas.concat(pieces)


@dataclass
class StreamPart:
    name: str
    start: int
    stop: int
    where: str | None = None


class ChunkReader:
    def __init__(self, file, chunk=DEFAULTS["chunk"]):
        self.file = file
        self.chunk = int(chunk)
        if self.chunk < 1:
            raise BuildError(f"stream.chunk must be a positive integer, got {chunk!r}")
        self.path = Path(str(file))
        self.suffix = self.path.suffix.lower()
        if self.suffix not in PARQUET and self.suffix not in SEPARATORS:
            raise BuildError(f"{self.path}: the table format {self.suffix!r} is unknown; use .parquet or .csv")
        self.names = None
        self.total = None

    def is_parquet(self):
        return self.suffix in PARQUET

    def header(self):
        if self.names is None:
            try:
                if self.is_parquet():
                    import pyarrow.parquet

                    with pyarrow.parquet.ParquetFile(str(self.path)) as handle:
                        names = list(handle.schema_arrow.names)
                else:
                    names = list(pandas.read_csv(self.path, nrows=0, sep=SEPARATORS[self.suffix]).columns)
            except (OSError, ValueError) as error:
                raise BuildError(f"{self.path}: cannot read the table header ({error})") from error
            self.names = [name for name in names if not str(name).startswith(INDEX_PREFIX)]
        return list(self.names)

    def count(self):
        if self.total is None:
            try:
                if self.is_parquet():
                    import pyarrow.parquet

                    with pyarrow.parquet.ParquetFile(str(self.path)) as handle:
                        self.total = int(handle.metadata.num_rows)
                else:
                    with pandas.read_csv(self.path, sep=SEPARATORS[self.suffix], usecols=[0],
                                         chunksize=self.chunk) as reader:
                        self.total = sum(len(frame) for frame in reader)
            except (OSError, ValueError) as error:
                raise BuildError(f"{self.path}: cannot count the table rows ({error})") from error
        return self.total

    def parquet_pieces(self, columns, start, stop):
        import pyarrow.parquet

        with pyarrow.parquet.ParquetFile(str(self.path)) as handle:
            meta = handle.metadata
            groups, first, offset = [], None, 0
            for index in range(meta.num_row_groups):
                size = meta.row_group(index).num_rows
                if size and offset + size > start and (stop is None or offset < stop):
                    first = offset if first is None else first
                    groups.append(index)
                offset += size
            if not groups:
                return
            position = first
            selected = self.header() if columns is None else columns
            for batch in handle.iter_batches(batch_size=self.chunk, row_groups=groups, columns=selected):
                yield position, batch.num_rows, batch
                position += batch.num_rows

    def csv_pieces(self, columns):
        with pandas.read_csv(self.path, sep=SEPARATORS[self.suffix], usecols=columns,
                             chunksize=self.chunk) as reader:
            position = 0
            for frame in reader:
                yield position, len(frame), frame
                position += len(frame)

    def pieces(self, columns, start, stop):
        if self.is_parquet():
            return self.parquet_pieces(columns, start, stop)
        return self.csv_pieces(columns)

    def chunks(self, columns=None, start=0, stop=None, shard=0, shards=1):
        start = max(int(start or 0), 0)
        stop = None if stop is None else int(stop)
        if stop is not None and stop <= start:
            return
        position = 0
        with closing(self.pieces(None if columns is None else list(columns), start, stop)) as pieces:
            for first, size, piece in pieces:
                last = first + size
                if not size or last <= start:
                    continue
                if stop is not None and first >= stop:
                    break
                owned = position % shards == shard
                position += 1
                if not owned:
                    continue
                frame = piece if isinstance(piece, pandas.DataFrame) else piece.to_pandas()
                frame.index = pandas.RangeIndex(first, last)
                low = max(start - first, 0)
                high = size if stop is None else min(stop - first, size)
                yield frame.iloc[low:high] if low or high < size else frame


class StreamDataset(IterableDataset):
    rows = None

    def __init__(self, engine, part, set_name):
        self.engine = engine
        self.part = part
        self.set_name = set_name
        self.count = engine.count(part)

    def generator(self):
        table = self.engine.table
        if self.set_name != "train" or table.rng is None or not table.batch.get("shuffle", True):
            return None
        return table.rng.stream("loader")

    def __iter__(self):
        table = self.engine.table
        info = get_worker_info()
        shard, shards = (0, 1) if info is None else (int(info.id), int(info.num_workers))
        generator = self.generator()
        with closing(self.engine.frames(self.part, shard=shard, shards=shards)) as frames:
            for frame in frames:
                if not len(frame):
                    continue
                x = table.features_of(frame)
                targets = {target.name: table.target_tensor(target, frame) for target in table.fitted_targets
                           if target.name in frame.columns}
                targets.update(table.weight_tensors(frame))
                if generator is None:
                    order = range(len(frame))
                else:
                    order = torch.randperm(len(frame), generator=generator).tolist()
                for position in order:
                    item = {"x": x[position]}
                    for name, values in targets.items():
                        item[name] = values[position]
                    yield item


def options_of(options):
    if options is True:
        return dict(DEFAULTS)
    if not isinstance(options, dict):
        raise BuildError(f"stream is true or a mapping of {list(DEFAULTS)}, got {options!r}")
    found = dict(DEFAULTS, **options)
    unknown = sorted(set(found) - set(DEFAULTS))
    if unknown:
        raise BuildError(f"stream: unknown keys {unknown}; the keys are {list(DEFAULTS)}")
    for key in DEFAULTS:
        positive_int(found[key], f"stream.{key}")
    return found


class StreamEngine:
    def __init__(self, table, options):
        self.table = table
        self.options = options_of(options)
        self.reader = None
        self.known = {}

    def sets_of(self, ctx):
        table = self.table
        split = table.split
        if not isinstance(split, (splits.SequentialSplit, splits.GivenSplit)):
            raise BuildError(f"data.split: {STREAM_SPLIT}")
        if getattr(split, "group", None) is not None:
            raise BuildError("data.split: a stream cuts a sequential split over the whole file and takes no group")
        if table.feed is not None:
            raise BuildError("data.stream: a stream yields single rows and takes no feed; drop stream or feed")
        self.reader = ChunkReader(locate(table.file, config_directories(ctx)), self.options["chunk"])
        self.known = {}
        parts = self.parts_of(split)
        train = parts.pop("train", None)
        if train is None or not self.count(train):
            raise BuildError("the split leaves no train rows")
        return {"train": train, **self.materialized(list(parts.values()))}

    def parts_of(self, split):
        count = self.reader.count()
        if isinstance(split, splits.SequentialSplit):
            parts, start = {}, 0
            for name, size in zip(splits.SET_NAMES, splits.sizes_of(count, split.ratios)):
                if size:
                    parts[name] = StreamPart(name, start, start + size)
                start += size
            return parts
        if split.column not in self.reader.header():
            raise BuildError(f"given split: the column {split.column!r} is not in the table")
        labels = {"train": split.train, "valid": split.valid, "test": split.test}
        return {name: StreamPart(name, 0, count, label) for name, label in labels.items() if label is not None}

    def selected(self, frame, part, weigh=True):
        table = self.table
        rows = frame.index
        if len(rows) and (rows[0] < part.start or rows[-1] >= part.stop):
            frame = frame[(rows >= part.start) & (rows < part.stop)]
        if part.where is not None:
            frame = frame[(frame[table.split.column] == part.where).to_numpy()]
        if table.all_filter:
            frame = query(frame, table.all_filter, "filter")
        text = table.set_filters.get(part.name)
        if text:
            frame = query(frame, text, f"filter.{part.name}")
        return table.dropped(table.weighed(frame) if weigh else frame)

    def frames(self, part, columns=None, shard=0, shards=1, weigh=True):
        with closing(self.reader.chunks(columns, part.start, part.stop, shard, shards)) as chunks:
            for frame in chunks:
                yield self.selected(frame, part, weigh)

    def materialized(self, parts):
        if not parts:
            return {}
        collected = {part.name: [] for part in parts}
        start = min(part.start for part in parts)
        stop = max(part.stop for part in parts)
        with closing(self.reader.chunks(None, start, stop)) as chunks:
            for frame in chunks:
                for part in parts:
                    piece = self.selected(frame, part)
                    if len(piece):
                        collected[part.name].append(piece)
        return {name: joined(pieces) for name, pieces in collected.items() if pieces}

    def fit_sample(self, sets):
        part = sets["train"]
        limit = self.options["fit_rows"]
        pieces, taken = [], 0
        with closing(self.frames(part)) as frames:
            for frame in frames:
                if not len(frame):
                    continue
                piece = frame.iloc[:limit - taken]
                pieces.append(piece)
                taken += len(piece)
                if taken >= limit:
                    break
        if not pieces:
            raise BuildError("the split leaves no train rows")
        return joined(pieces)

    def dataset(self, part, set_name):
        return StreamDataset(self, part, set_name)

    def count(self, part):
        if isinstance(part, pandas.DataFrame):
            return len(part)
        key = (part.name, part.start, part.stop, part.where)
        if key not in self.known:
            filtered = bool(self.table.all_filter or self.table.set_filters.get(part.name))
            if part.where is None and not filtered:
                self.known[key] = max(part.stop - part.start, 0)
            else:
                columns = None if filtered else [self.table.split.column]
                with closing(self.frames(part, columns, weigh=False)) as frames:
                    self.known[key] = sum(len(frame) for frame in frames)
        return self.known[key]
