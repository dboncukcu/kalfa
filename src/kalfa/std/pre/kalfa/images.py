import torch

from kalfa.registration import lego
from kalfa.std.common.rng import Draws
from kalfa.std.pre.base import ImageTensor, Preprocessor, pil_image


@lego("/pre/kalfa/resize", alias="resize", description="Resize an image to size (int or [h, w])")
class Resize(Preprocessor):
    def __init__(self, size):
        self.size = (int(size), int(size)) if isinstance(size, (int, float)) else (int(size[1]), int(size[0]))

    def apply(self, value):
        return pil_image(value).resize(self.size)


@lego("/pre/kalfa/to_tensor", alias="to_tensor",
      description="Image to a float tensor in [0, 1], channels first")
class ToTensor(ImageTensor):
    signed = False


@lego("/pre/kalfa/to_tensor_signed", alias="to_tensor_signed",
      description="Image to a float tensor in [-1, 1], channels first")
class ToTensorSigned(ImageTensor):
    signed = True


@lego("/pre/kalfa/normalize", alias="normalize",
      description="Normalize an image tensor per channel; mean and std are numbers, lists or the presets "
                  "imagenet and cifar10")
class Normalize(Preprocessor):
    dtype = "float32"
    presets = {"imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
               "cifar10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])}

    def __init__(self, mean, std):
        self.mean = self.presets[mean][0] if isinstance(mean, str) else mean
        self.std = self.presets[std][1] if isinstance(std, str) else std

    def apply(self, value):
        channels = value.shape[0]
        mean = torch.as_tensor(self.mean if isinstance(self.mean, list) else [self.mean] * channels,
                               dtype=value.dtype).reshape(-1, 1, 1)
        std = torch.as_tensor(self.std if isinstance(self.std, list) else [self.std] * channels,
                              dtype=value.dtype).reshape(-1, 1, 1)
        return (value - mean) / std


@lego("/pre/kalfa/random_crop_flip", alias="random_crop_flip",
      description="Random crop of size after padding and a random horizontal flip")
class RandomCropFlip(Preprocessor):
    padding = 4

    def __init__(self, size):
        self.size = int(size)
        self.draws = Draws("random_crop_flip")

    def apply(self, value):
        from PIL import Image, ImageOps

        image = ImageOps.expand(pil_image(value), border=self.padding, fill=0)
        width, height = image.size
        draws = self.draws.numpy()
        left = int(draws.integers(0, max(width - self.size, 0) + 1))
        top = int(draws.integers(0, max(height - self.size, 0) + 1))
        image = image.crop((left, top, left + self.size, top + self.size))
        if draws.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image


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


@lego("/pre/kalfa/two_views", alias="two_views", refs={"transform": "preprocessor"},
      description="Two independent applications of a transform to one image, as a pair")
class TwoViews(Preprocessor):
    def __init__(self, transform):
        self.transform = transform

    def apply(self, value):
        return (self.transform.apply(value), self.transform.apply(value))
