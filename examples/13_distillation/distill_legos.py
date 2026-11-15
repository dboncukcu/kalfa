from torch import nn


class Classifier(nn.Module):
    def __init__(self, n_classes, width=16, channels=3):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                                      nn.Flatten())
        self.head = nn.Linear(width, n_classes)

    def forward(self, image):
        return self.head(self.features(image))
