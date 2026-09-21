import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb

class CondUNet(nn.Module):
    def __init__(self, num_classes=10, ch=64, t_dim=128, c_dim=32):
        super().__init__()
        self.t_dim = t_dim
        self.ch = ch
        self.c_emb = nn.Embedding(num_classes, c_dim)

        self.conv1 = nn.Conv2d(1, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch * 2, 4, 2, 1)   # 14
        self.conv3 = nn.Conv2d(ch * 2, ch * 2, 3, padding=1)
        self.conv4 = nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)  # 7
        self.conv5 = nn.Conv2d(ch * 4, ch * 4, 3, padding=1)

        self.t_mlp = nn.Sequential(nn.Linear(t_dim, ch * 4), nn.SiLU(), nn.Linear(ch * 4, ch * 4))

        self.up1 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1)  # 14
        self.up2 = nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1)      # 28
        self.out = nn.Conv2d(ch, 1, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch * 2)
        self.norm3 = nn.GroupNorm(8, ch * 2)
        self.norm4 = nn.GroupNorm(8, ch * 4)
        self.norm5 = nn.GroupNorm(8, ch * 4)

    def forward(self, x, t, y):
        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.t_mlp(t_emb)  # (B, ch*4)
        c_emb = self.c_emb(y)      # (B, c_dim)

        h1 = F.silu(self.norm1(self.conv1(x)))
        h2 = F.silu(self.norm2(self.conv2(h1)))
        h3 = F.silu(self.norm3(self.conv3(h2)))
        h4 = F.silu(self.norm4(self.conv4(h3)))
        h5 = F.silu(self.norm5(self.conv5(h4)))

        h5 = h5 + t_emb[:, :, None, None] + c_emb[:, :, None, None].repeat(1, h5.size(1) // c_emb.size(1), 1, 1) \
             if c_emb.size(1) < h5.size(1) else h5 + t_emb[:, :, None, None]

        d1 = F.silu(self.up1(h5))
        d2 = F.silu(self.up2(d1))
        return self.out(d2)

class DDPM:
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.model = model
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        a_bar = self.alpha_bars[t][:, None, None, None]
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

    def loss(self, x0, y):
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=self.device).long()
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.model(x_t, t.float(), y)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, y, shape=None):
        self.model.eval()
        B = y.size(0)
        if shape is None:
            shape = (B, 1, 28, 28)
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.T)):
            t = torch.full((B,), i, device=self.device, dtype=torch.float)
            pred_noise = self.model(x, t, y)
            alpha = self.alphas[i]
            alpha_bar = self.alpha_bars[i]
            beta = self.betas[i]
            mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise)
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean
        return x.clamp(0, 1)


def train_ddpm(ddpm, train_loader, epochs, lr, device):
    opt = torch.optim.Adam(ddpm.model.parameters(), lr=lr)
    history = []
    for ep in range(epochs):
        ddpm.model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ddpm.loss(x, y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        total /= len(train_loader.dataset)
        history.append(total)
    return history