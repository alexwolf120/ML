import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

from config import DEVICE, EPOCHS_AE, EPOCHS_DIFF, LR, LATENT_DIM, NUM_CLASSES, MODEL_CHOICE
from data_utils import get_dataloaders
from autoencoder import AE, train_ae
from generative_models import GaussianImageModel, SMOTEModel, make_class_mixture
from diffusion import CondUNet, DDPM, train_ddpm

OUT = "results"

def show_grid(imgs, title, path, n=8, cmap="gray"):
    imgs = imgs.detach().cpu().numpy() if torch.is_tensor(imgs) else imgs
    imgs = imgs.reshape(-1, 28, 28)
    n = min(n, len(imgs))
    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 2))
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(imgs[i], cmap=cmap)
        axes[i].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_history(hist, title, path):
    plt.figure(figsize=(6, 4))
    for k, v in hist.items():
        plt.plot(v, label=k)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    train_loader, val_loader, test_loader = get_dataloaders()
    ae = AE(latent_dim=LATENT_DIM).to(DEVICE)
    hist_ae = train_ae(ae, train_loader, val_loader, EPOCHS_AE, LR, DEVICE)
    plot_history(hist_ae, "AE loss", f"{OUT}/ae_loss.png")

    ae.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x.to(DEVICE)
        x_hat, _ = ae(x)
    show_grid(x[:8], "Original", f"{OUT}/ae_original.png")
    show_grid(x_hat[:8], "Reconstructed", f"{OUT}/ae_reconstructed.png")

    N_SUB = 3000
    xs, ys = [], []
    total = 0
    for x, y in train_loader:
        xs.append(x); ys.append(y)
        total += len(x)
        if total >= N_SUB:
            break
    X_train = torch.cat(xs)[:N_SUB]
    y_train = torch.cat(ys)[:N_SUB].numpy()

    with torch.no_grad():
        Z_train = ae.encode(X_train.to(DEVICE)).cpu().numpy()

    gmm_img = GaussianImageModel(n_components=1, latent=False)
    gmm_img.fit(X_train.numpy(), y_train)
    y_syn = np.arange(NUM_CLASSES)
    X_gmm_img, _ = gmm_img.sample(y_syn, n_per_class=1)
    show_grid(X_gmm_img, "GMM samples (image space)", f"{OUT}/gmm_image.png")

    gmm_lat = GaussianImageModel(n_components=1, latent=True)
    gmm_lat.fit(Z_train, y_train)
    Z_gmm, _ = gmm_lat.sample(y_syn, n_per_class=1)
    with torch.no_grad():
        X_gmm_lat = ae.decode(torch.tensor(Z_gmm, dtype=torch.float32).to(DEVICE))
    show_grid(X_gmm_lat, "GMM samples (latent -> AE decode)", f"{OUT}/gmm_latent.png")

    smote_img = SMOTEModel(k_neighbors=5)
    smote_img.fit_resample(X_train.numpy(), y_train, target_per_class=800)
    X_smote_img = np.stack([smote_img.sample_class(c, 1)[0] for c in y_syn])
    show_grid(X_smote_img, "SMOTE samples (image space)", f"{OUT}/smote_image.png")

    smote_lat = SMOTEModel(k_neighbors=5, latent=True)
    smote_lat.fit_resample(Z_train, y_train, target_per_class=800)
    Z_smote = np.stack([smote_lat.sample_class(c, 1)[0] for c in y_syn])
    with torch.no_grad():
        X_smote_lat = ae.decode(torch.tensor(Z_smote, dtype=torch.float32).to(DEVICE))
    show_grid(X_smote_lat, "SMOTE samples (latent -> AE decode)", f"{OUT}/smote_latent.png")

    unet = CondUNet(num_classes=NUM_CLASSES, ch=64).to(DEVICE)
    ddpm = DDPM(unet, T=500, device=DEVICE)
    hist_ddpm = train_ddpm(ddpm, train_loader, EPOCHS_DIFF, LR, DEVICE)
    plot_history({"loss": hist_ddpm}, "DDPM loss", f"{OUT}/ddpm_loss.png")

    y_cond = torch.arange(NUM_CLASSES, device=DEVICE)
    samples_cond = ddpm.sample(y_cond)
    show_grid(samples_cond, "DDPM conditional samples", f"{OUT}/ddpm_conditional.png")

    mix_classes = [0, 3, 7]
    weights = np.array([0.5, 0.3, 0.2])
    y_mix_np = np.random.choice(mix_classes, size=8, p=weights)
    y_mix = torch.tensor(y_mix_np, device=DEVICE)
    samples_mix = ddpm.sample(y_mix)
    show_grid(samples_mix, f"DDPM mixture {list(zip(y_mix_np, np.round(weights, 2)))}",
              f"{OUT}/ddpm_mixture.png")

if __name__ == "__main__":
    freeze_support()
    main()