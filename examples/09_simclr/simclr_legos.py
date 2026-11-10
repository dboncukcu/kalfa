from torch import nn


class Encoder(nn.Module):
    def __init__(self, width=32, channels=3):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(16, width))

    def forward(self, image):
        if image.ndim == 5:
            image = image.flatten(0, 1)
        return self.net(image)
