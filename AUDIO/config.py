import os

DATA_DIR = "./data"
OUTPUT_DIR = "./output"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 5.0
N_MFCC = 40
RANDOM_STATE = 42
TSNE_PERPLEXITY = 30
AUDIO_COLUMN = "audio"
LABEL_COLUMN = "nombreCientifico"
