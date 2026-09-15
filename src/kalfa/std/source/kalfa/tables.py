import pandas

from kalfa.std.common.log import clock, logger_for, since
from kalfa.std.common.stream import CsvChunks, ParquetChunks, Stream


logger = logger_for("data.source")


def parquet(path, columns=None):
    logger.info(f"reading {path}")
    started = clock()
    df = pandas.read_parquet(path, columns=None if columns is None else list(columns))
    logger.info(f"{len(df)} rows, {len(df.columns)} columns ({since(started)})")
    return df


def csv(path, columns=None):
    logger.info(f"reading {path}")
    started = clock()
    df = pandas.read_csv(path, usecols=None if columns is None else list(columns))
    logger.info(f"{len(df)} rows, {len(df.columns)} columns ({since(started)})")
    return df


def parquet_stream(path, chunk=65536, columns=None):
    logger.info(f"streaming {path} in chunks of {chunk} rows")
    return Stream(ParquetChunks(path, chunk, columns))


def csv_stream(path, chunk=65536, columns=None):
    logger.info(f"streaming {path} in chunks of {chunk} rows")
    return Stream(CsvChunks(path, chunk, columns))
