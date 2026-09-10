import pandas

from kalfa.registration import lego
from kalfa.std.common.log import clock, logger_for, since
from kalfa.std.common.stream import CsvChunks, ParquetChunks, Stream


logger = logger_for("data.source")


@lego("/source/kalfa/parquet", returns="df", alias="parquet", header="/lego/kalfa/parquet_header",
      description="Read a parquet file into a DataFrame; columns lists the ones to read, the others stay on disk")
def parquet(path, columns=None):
    logger.info(f"reading {path}")
    started = clock()
    df = pandas.read_parquet(path, columns=None if columns is None else list(columns))
    logger.info(f"{len(df)} rows, {len(df.columns)} columns ({since(started)})")
    return df


@lego("/source/kalfa/csv", returns="df", alias="csv", header="/lego/kalfa/csv_header",
      description="Read a CSV file into a DataFrame; columns lists the ones to read, the others stay on disk")
def csv(path, columns=None):
    logger.info(f"reading {path}")
    started = clock()
    df = pandas.read_csv(path, usecols=None if columns is None else list(columns))
    logger.info(f"{len(df)} rows, {len(df.columns)} columns ({since(started)})")
    return df


@lego("/source/kalfa/parquet_stream", returns="df", header="/lego/kalfa/parquet_header", stream=True,
      description="Read a parquet file in chunks (the lazy set): a stream the data legos filter, cut and fit "
                  "without loading the table")
def parquet_stream(path, chunk=65536, columns=None):
    logger.info(f"streaming {path} in chunks of {chunk} rows")
    return Stream(ParquetChunks(path, chunk, columns))


@lego("/source/kalfa/csv_stream", returns="df", header="/lego/kalfa/csv_header", stream=True,
      description="Read a CSV file in chunks (the lazy set)")
def csv_stream(path, chunk=65536, columns=None):
    logger.info(f"streaming {path} in chunks of {chunk} rows")
    return Stream(CsvChunks(path, chunk, columns))
