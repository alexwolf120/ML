import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 368528
IMG_SIZE = 28
IMG_CH = 1
NUM_CLASSES = 10
LATENT_DIM = 32
BATCH_SIZE = 128
EPOCHS_AE = 15
EPOCHS_DIFF = 30
LR = 1e-3
MODEL_CHOICE = 2

torch.manual_seed(SEED)