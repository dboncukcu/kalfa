from kalfa.registration import lego
from kalfa.std.common.log import logger_for
from kalfa.std.common.samples import Samples
from kalfa.std.source.base import TextLines


logger = logger_for("data.source")


@lego("/source/kalfa/text_lines", returns="df", alias="text_lines", header="/lego/kalfa/text_lines_header",
      samples=True,
      description="The lines of a text file as a Dataset with the field text")
def text_lines(path):
    logger.info(f"reading {path}")
    source = TextLines(path)
    logger.info(f"{len(source)} lines")
    return Samples(source)
