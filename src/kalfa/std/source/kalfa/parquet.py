from kalfa.registration import lego
from kalfa.std.common.log import clock, logger_for, since
import pandas


logger = logger_for("data.source")


@lego("/source/kalfa/parquet", returns="df", alias="parquet", header="/lego/kalfa/parquet_header",
      description="Read a parquet file into a DataFrame")
def parquet(path):
    logger.info(f"reading {path}")
    started = clock()
    df = pandas.read_parquet(path)
    logger.info(f"{len(df)} rows, {len(df.columns)} columns ({since(started)})")
    return df
