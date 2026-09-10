from kalfa.registration import lego
from kalfa.std.common.rng import Draws
from kalfa.std.pre.base import Preprocessor, pil_image


@lego("/pre/kalfa/simclr_aug", alias="simclr_aug",
      description="SimCLR augmentation: random resized crop to size, horizontal flip, brightness jitter")
class SimclrAug(Preprocessor):
    def __init__(self, size, scale=(0.5, 1.0)):
        self.size = int(size)
        self.scale = tuple(scale)
        self.draws = Draws("simclr_aug")

    def apply(self, value):
        from PIL import Image, ImageEnhance

        image = pil_image(value)
        width, height = image.size
        draws = self.draws.numpy()
        fraction = float(draws.uniform(self.scale[0], self.scale[1]))
        crop_width = max(1, int(round(width * fraction)))
        crop_height = max(1, int(round(height * fraction)))
        left = int(draws.integers(0, width - crop_width + 1))
        top = int(draws.integers(0, height - crop_height + 1))
        image = image.crop((left, top, left + crop_width, top + crop_height)).resize((self.size, self.size))
        if draws.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return ImageEnhance.Brightness(image).enhance(float(draws.uniform(0.6, 1.4)))
