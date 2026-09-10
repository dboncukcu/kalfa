from kalfa.registration import lego
from kalfa.std.common.log import logger_for
from kalfa.std.frame.base import read_frames, write_frames


logger = logger_for("data.frames")


@lego("/lego/kalfa/transform_set",
      description="Apply the transforms that name this set, in order; the frame passes untouched without any")
def transform_set(df, set, transforms):
    out = df
    for transform in transforms or []:
        out = transform(out)
    return out


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


@lego("/lego/kalfa/apply_frames",
      description="Apply the fitted frame transforms to one set, in the order they were fitted")
def apply_frames(df, frames):
    table = df
    for transform in frames or []:
        table = transform.apply(table)
    return table


@lego("/lego/kalfa/read_frames", returns="frames",
      description="The fitted frame transforms of a record, read from fitted/frames")
def frames_of(record):
    return read_frames(record)
