import torch
from torch import nn


class Vae(nn.Module):
    def __init__(self, latent=32, channels=1, width=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, width, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten())
        self.mu = nn.Linear(width * 16, latent)
        self.logvar = nn.Linear(width * 16, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, width * 16), nn.ReLU(), nn.Unflatten(1, (width, 4, 4)),
            nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(width, channels, 4, stride=2, padding=1))

    def forward(self, image):
        hidden = self.encoder(image)
        mu, logvar = self.mu(hidden), self.logvar(hidden)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.decoder(z), mu, logvar
