import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LATENT_DIM, IMG_CH

class AE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(IMG_CH, 32, 4, 2, 1),   # 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),       # 7
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),      # 7
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(128 * 7 * 7, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 128 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, IMG_CH, 4, 2, 1),  # 28
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 128, 7, 7)
        return self.dec(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def train_ae(model, train_loader, val_loader, epochs, lr, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train": [], "val": []}
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, _ = model(x)
            loss = F.mse_loss(x_hat, x)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_hat, _ = model(x)
                val_loss += F.mse_loss(x_hat, x).item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        history["train"].append(tr_loss)
        history["val"].append(val_loss)
    return history