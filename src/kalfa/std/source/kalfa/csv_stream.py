from kalfa.registration import lego
from kalfa.std.common.log import logger_for
from kalfa.std.common.stream import CsvChunks, Stream


logger = logger_for("data.source")


@lego("/source/kalfa/csv_stream", returns="df", header="/lego/kalfa/csv_header", stream=True,
      description="Read a CSV file in chunks (the lazy set)")
def csv_stream(path, chunk=65536):
    logger.info(f"streaming {path} in chunks of {chunk} rows")
    return Stream(CsvChunks(path, chunk))
