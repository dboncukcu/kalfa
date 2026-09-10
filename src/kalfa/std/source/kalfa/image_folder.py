from kalfa.registration import lego
from kalfa.std.common.log import logger_for
from kalfa.std.common.samples import Samples
from kalfa.std.source.base import ImageFolder


logger = logger_for("data.source")


@lego("/source/kalfa/image_folder", returns="df", alias="image_folder",
      description="Images under root/<class>/ as a Dataset with fields image and label")
def image_folder(path):
    logger.info(f"reading {path}")
    folder = ImageFolder(path)
    logger.info(f"{len(folder)} images in {len(folder.classes)} classes")
    return Samples(folder)
