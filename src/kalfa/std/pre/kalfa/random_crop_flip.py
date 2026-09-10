import numpy

from kalfa.registration import lego
from kalfa.std.pre.base import Preprocessor, pil_image


@lego("/pre/kalfa/random_crop_flip", alias="random_crop_flip",
      description="Random crop of size after padding and a random horizontal flip")
class RandomCropFlip(Preprocessor):
    padding = 4

    def __init__(self, size):
        self.size = int(size)

    def apply(self, value):
        from PIL import Image, ImageOps

        image = ImageOps.expand(pil_image(value), border=self.padding, fill=0)
        width, height = image.size
        left = int(numpy.random.randint(0, max(width - self.size, 0) + 1))
        top = int(numpy.random.randint(0, max(height - self.size, 0) + 1))
        image = image.crop((left, top, left + self.size, top + self.size))
        if numpy.random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image
