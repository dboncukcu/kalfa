import torch


def pixel_features(size=4):
    def extract(images):
        return torch.nn.functional.adaptive_avg_pool2d(images, int(size)).flatten(1)

    return extract
