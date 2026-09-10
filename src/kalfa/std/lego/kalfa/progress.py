from kalfa.registration import lego
from kalfa.std.common.log import Progress


@lego("/lego/kalfa/progress", description="The progress display of a run")
def progress():
    return Progress()
