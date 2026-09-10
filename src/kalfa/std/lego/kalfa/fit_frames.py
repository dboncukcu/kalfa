from kalfa.registration import lego
from kalfa.std.common.log import logger_for
from kalfa.std.frame.base import write_frames


logger = logger_for("data.frames")


@lego("/lego/kalfa/fit_frames", returns="frames", state=True, bus=["record"],
      description="Fit the frame transforms on the train set, each on what the ones before it produced, and keep "
                  "them in the record under fitted/frames")
def fit_frames(df, frames, record=None):
    fitted = []
    table = df
    for transform in frames or []:
        transform.fit(table)
        table = transform.apply(table)
        fitted.append(transform)
    if fitted:
        logger.info(f"fitted {len(fitted)} frame transforms on the train set")
    if record is not None:
        write_frames(fitted, record)
    return fitted
