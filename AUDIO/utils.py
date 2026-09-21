# utils.py
import io
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datasets import load_dataset, Audio
from config import (SAMPLE_RATE, DURATION, N_MFCC, RANDOM_STATE,
                    TSNE_PERPLEXITY, AUDIO_COLUMN, LABEL_COLUMN, OUTPUT_DIR)

def _detect_columns(dataset):
    audio_col, label_col = AUDIO_COLUMN, LABEL_COLUMN
    cols = dataset.column_names

    if audio_col is None:
        for cand in ["audio", "file", "path", "sound", "wav", "archivo"]:
            if cand in cols:
                audio_col = cand
                break
        if audio_col is None:
            raise ValueError(f"Не удалось найти колонку с аудио. Доступные: {cols}")

    if label_col is None:
        for cand in ["label", "labels", "target", "class", "category",
                     "bird", "species", "nombreCientifico", "nombre",
                     "cientifico", "clase", "categoria"]:
            if cand in cols:
                label_col = cand
                break
        if label_col is None:
            remaining = [c for c in cols if c != audio_col]
            if len(remaining) == 1:
                label_col = remaining[0]
            else:
                raise ValueError(f"Не удалось найти колонку с меткой. Доступные: {cols}")

    return audio_col, label_col


def load_audio_dataset(dataset_name="capa2000/binary-classifier-birdnet"):
    ds = load_dataset(dataset_name)

    if "test" not in ds:
        split = ds["train"].train_test_split(test_size=0.2, seed=RANDOM_STATE)
        train_ds, test_ds = split["train"], split["test"]
    else:
        train_ds, test_ds = ds["train"], ds["test"]

    audio_col, label_col = _detect_columns(train_ds)

    train_ds = train_ds.cast_column(audio_col, Audio(decode=False))
    test_ds = test_ds.cast_column(audio_col, Audio(decode=False))

    rename_map = {audio_col: "audio", label_col: "label"}
    if audio_col != "audio" or label_col != "label":
        train_ds = train_ds.rename_columns(rename_map)
        test_ds = test_ds.rename_columns(rename_map)

    all_labels = list(train_ds["label"]) + list(test_ds["label"])
    if isinstance(all_labels[0], str):
        le = LabelEncoder()
        le.fit(all_labels)
        train_ds = train_ds.map(lambda x: {"label": int(le.transform([x["label"]])[0])})
        test_ds = test_ds.map(lambda x: {"label": int(le.transform([x["label"]])[0])})
    return train_ds, test_ds


def _decode_audio(example):
    audio = example["audio"]
    if isinstance(audio, dict) and audio.get("bytes") is not None:
        array, sr = sf.read(io.BytesIO(audio["bytes"]))
    elif isinstance(audio, dict) and audio.get("array") is not None:
        array, sr = audio["array"], audio["sampling_rate"]
    elif isinstance(audio, (bytes, bytearray)):
        array, sr = sf.read(io.BytesIO(audio))
    else:
        raise ValueError(f"Неизвестный формат аудио: {type(audio)}")

    if array.ndim > 1:
        array = array.mean(axis=1)
    return array.astype(np.float32), sr


def extract_mfcc(audio_array, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])


def _prepare_audio(audio, sr):
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    target_len = int(SAMPLE_RATE * DURATION)
    if len(audio) > target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))
    return audio


def extract_mfcc_dataset(dataset):
    X, y = [], []
    for example in dataset:
        audio, sr = _decode_audio(example)
        audio = _prepare_audio(audio, sr)
        X.append(extract_mfcc(audio))
        y.append(example["label"])
    return np.array(X), np.array(y)


def visualize_features(X, y, title="Feature Space", method="tsne"):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    reducer = (TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=RANDOM_STATE)
               if method == "tsne"
               else PCA(n_components=2, random_state=RANDOM_STATE))
    X_2d = reducer.fit_transform(X_scaled)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Class")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{title.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(fname)
    plt.show()


def find_extreme_pairs(X, y, n_pairs=2):
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(X, metric="euclidean"))
    np.fill_diagonal(dist_matrix, np.inf)

    flat_sorted = np.argsort(dist_matrix, axis=None)
    closest, farthest = [], []
    seen = set()
    for idx in flat_sorted:
        i, j = np.unravel_index(idx, dist_matrix.shape)
        if i < j and (i, j) not in seen:
            closest.append((i, j, float(dist_matrix[i, j])))
            seen.add((i, j))
        if len(closest) == n_pairs:
            break

    seen = set()
    for idx in flat_sorted[::-1]:
        i, j = np.unravel_index(idx, dist_matrix.shape)
        if i < j and (i, j) not in seen:
            farthest.append((i, j, float(dist_matrix[i, j])))
            seen.add((i, j))
        if len(farthest) == n_pairs:
            break

    return closest, farthest


def plot_spectrogram(audio, sr, title="Spectrogram", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax)
    ax.set_title(title)
    return ax