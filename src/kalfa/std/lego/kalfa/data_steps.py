from kalfa.std.common.log import logger_for
from kalfa.std.frame.base import read_frames, write_frames


logger = logger_for("data.frames")


def transform_set(df, set, transforms):
    out = df
    for transform in transforms or []:
        out = transform(out)
    return out


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


def apply_frames(df, frames):
    table = df
    for transform in frames or []:
        table = transform.apply(table)
    return table


def frames_of(record):
    return read_frames(record)
