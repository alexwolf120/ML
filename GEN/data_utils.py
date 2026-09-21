import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from config import BATCH_SIZE, IMG_SIZE, SEED


def get_dataloaders(batch_size=BATCH_SIZE, val_size=0.1):
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    train_full = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=tfm)

    idx = np.arange(len(train_full))
    tr_idx, val_idx = train_test_split(idx, test_size=val_size, random_state=SEED, stratify=train_full.targets)

    train_ds = Subset(train_full, tr_idx)
    val_ds = Subset(train_full, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader