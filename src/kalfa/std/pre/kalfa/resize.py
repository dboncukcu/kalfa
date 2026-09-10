from kalfa.registration import lego
from kalfa.std.pre.base import Preprocessor, pil_image


@lego("/pre/kalfa/resize", alias="resize", description="Resize an image to size (int or [h, w])")
class Resize(Preprocessor):
    def __init__(self, size):
        self.size = (int(size), int(size)) if isinstance(size, (int, float)) else (int(size[1]), int(size[0]))

    def apply(self, value):
        return pil_image(value).resize(self.size)
