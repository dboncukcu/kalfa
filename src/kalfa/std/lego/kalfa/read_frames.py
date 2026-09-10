from kalfa.registration import lego
from kalfa.std.frame.base import read_frames as read_fitted


@lego("/lego/kalfa/read_frames", returns="frames",
      description="The fitted frame transforms of a record, read from fitted/frames")
def read_frames(record):
    return read_fitted(record)
