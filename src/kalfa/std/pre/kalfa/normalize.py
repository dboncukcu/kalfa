from kalfa.registration import lego
from kalfa.std.pre.base import Preprocessor


IMAGE_STATS = {"imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
               "cifar10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])}


class Normalize(Preprocessor):
    dtype = "float32"

    def __init__(self, mean, std):
        self.mean = IMAGE_STATS[mean][0] if isinstance(mean, str) else mean
        self.std = IMAGE_STATS[std][1] if isinstance(std, str) else std

    def apply(self, value):
        import torch

        channels = value.shape[0]
        mean = torch.as_tensor(self.mean if isinstance(self.mean, list) else [self.mean] * channels,
                               dtype=value.dtype).reshape(-1, 1, 1)
        std = torch.as_tensor(self.std if isinstance(self.std, list) else [self.std] * channels,
                              dtype=value.dtype).reshape(-1, 1, 1)
        return (value - mean) / std


@lego("/pre/kalfa/normalize", alias="normalize",
      description="Normalize an image tensor per channel; mean and std are numbers, lists or the presets "
                  "imagenet and cifar10")
def normalize(mean, std):
    return Normalize(mean, std)
