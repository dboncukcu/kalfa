from kalfa.registration import lego
from kalfa.std.common.log import clock, logger_for, since


logger = logger_for("data.source")


@lego("/source/kalfa/csv", returns="df", alias="csv",
      description="Read a CSV file into a DataFrame")
def csv(path):
    import pandas

    logger.info(f"reading {path}")
    started = clock()
    df = pandas.read_csv(path)
    logger.info(f"{len(df)} rows, {len(df.columns)} columns ({since(started)})")
    return df
