import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, latent, n_classes, channels=3, width=8, embed=8):
        super().__init__()
        self.width = width
        self.embed = nn.Embedding(n_classes, embed)
        self.project = nn.Linear(latent + embed, width * 2 * 16)
        self.body = nn.Sequential(
            nn.ReLU(), nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1),
            nn.ReLU(), nn.ConvTranspose2d(width, channels, 4, stride=2, padding=1), nn.Tanh())

    def forward(self, z, label):
        hidden = self.project(torch.cat([z, self.embed(label)], dim=1))
        return self.body(hidden.view(-1, self.width * 2, 4, 4))


class Critic(nn.Module):
    def __init__(self, n_classes, channels=3, width=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, width, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(width, width * 2, 4, stride=2, padding=1), nn.LeakyReLU(0.2), nn.Flatten())
        self.score = nn.Linear(width * 2 * 16, 1)
        self.embed = nn.Embedding(n_classes, width * 2 * 16)

    def forward(self, image, label):
        features = self.features(image)
        return self.score(features) + (self.embed(label) * features).sum(dim=1, keepdim=True)
