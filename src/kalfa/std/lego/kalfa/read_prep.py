from kalfa.registration import lego
from kalfa.std.pre.base import read_prep as read_fitted


@lego("/lego/kalfa/read_prep", returns="prep",
      description="The fitted preprocessing plan of a record, read from its preprocessors directory")
def read_prep(record):
    return read_fitted(record)
