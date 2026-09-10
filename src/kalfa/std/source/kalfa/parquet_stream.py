from kalfa.registration import lego
from kalfa.std.common.log import logger_for
from kalfa.std.common.stream import ParquetChunks, Stream


logger = logger_for("data.source")


@lego("/source/kalfa/parquet_stream", returns="df",
      description="Read a parquet file in chunks (the lazy set): a stream the data legos filter, cut and fit "
                  "without loading the table")
def parquet_stream(path, chunk=65536):
    logger.info(f"streaming {path} in chunks of {chunk} rows")
    return Stream(ParquetChunks(path, chunk))
