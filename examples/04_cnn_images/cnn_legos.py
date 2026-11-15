from torch import nn


class Backbone(nn.Module):
    def __init__(self, width=16, channels=3):
        super().__init__()
        self.out_features = 2 * width
        self.features = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(width, 2 * width, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                                      nn.Flatten())

    def forward(self, image):
        return self.features(image)


class Classifier(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.out_features, n_classes)

    def forward(self, image):
        return self.head(self.backbone(image))
