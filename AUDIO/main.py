import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, SAMPLE_RATE
from utils import (load_audio_dataset, extract_mfcc_dataset, visualize_features,
                   find_extreme_pairs, plot_spectrogram, _decode_audio,
                   _prepare_audio)
from classic import train_classic_models

train_ds, test_ds = load_audio_dataset()

num_classes = len(set(train_ds["label"]) | set(test_ds["label"]))
X_train_mfcc, y_train = extract_mfcc_dataset(train_ds)
X_test_mfcc, y_test = extract_mfcc_dataset(test_ds)
visualize_features(X_train_mfcc, y_train, title="MFCC Feature Space (t-SNE)")
results_mfcc, models_mfcc, scaler_mfcc = train_classic_models(
    X_train_mfcc, y_train, X_test_mfcc, y_test
)
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
pretrained_model = AutoModel.from_pretrained("facebook/wav2vec2-base")
pretrained_model.eval()


def _dataset_to_arrays(dataset):
    audios, labels = [], []
    for ex in dataset:
        audio, sr = _decode_audio(ex)
        audio = _prepare_audio(audio, sr)
        audios.append(audio)
        labels.append(ex["label"])
    return audios, labels


train_audios, train_labels = _dataset_to_arrays(train_ds)
test_audios, test_labels = _dataset_to_arrays(test_ds)


def extract_deep_features(audios, batch_size=8):
    X = []
    for i in range(0, len(audios), batch_size):
        batch = audios[i:i + batch_size]
        inputs = feature_extractor(batch, sampling_rate=SAMPLE_RATE,
                                   return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = pretrained_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        X.append(emb)
    return np.vstack(X)

X_train_deep = extract_deep_features(train_audios)
X_test_deep = extract_deep_features(test_audios)
y_train_deep = np.array(train_labels)
y_test_deep = np.array(test_labels)

visualize_features(X_train_deep, y_train_deep, title="Deep Feature Space (t-SNE)")

results_deep, models_deep, scaler_deep = train_classic_models(
    X_train_deep, y_train_deep, X_test_deep, y_test_deep
)

hidden_dim = X_train_deep.shape[1]
classifier_head = nn.Sequential(
    nn.Linear(hidden_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, num_classes),
)

for p in pretrained_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(classifier_head.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

X_train_tensor = torch.tensor(X_train_deep, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_deep, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_deep, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_deep, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                          batch_size=16, shuffle=True)

for epoch in range(10):
    classifier_head.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(classifier_head(xb), yb)
        loss.backward()
        optimizer.step()

classifier_head.eval()
with torch.no_grad():
    preds = classifier_head(X_test_tensor).argmax(dim=1).numpy()
acc_head = accuracy_score(y_test_deep, preds)

for p in pretrained_model.parameters():
    p.requires_grad = True


class AudioClassifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, input_values):
        out = self.encoder(input_values).last_hidden_state.mean(dim=1)
        return self.head(out)


model = AudioClassifier(pretrained_model, classifier_head)
optimizer_ft = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion_ft = nn.CrossEntropyLoss()


def collate_audio(batch_indices):
    audios = [train_audios[i] for i in batch_indices]
    labels = torch.tensor([train_labels[i] for i in batch_indices], dtype=torch.long)
    inputs = feature_extractor(audios, sampling_rate=SAMPLE_RATE,
                               return_tensors="pt", padding=True)
    return inputs.input_values, labels


train_loader_ft = DataLoader(range(len(train_audios)), batch_size=4,
                             shuffle=True, collate_fn=collate_audio)

for epoch in range(3):
    model.train()
    for input_values, labels in train_loader_ft:
        optimizer_ft.zero_grad()
        loss = criterion_ft(model(input_values), labels)
        loss.backward()
        optimizer_ft.step()


def collate_audio_test(batch_indices):
    audios = [test_audios[i] for i in batch_indices]
    labels = torch.tensor([test_labels[i] for i in batch_indices], dtype=torch.long)
    inputs = feature_extractor(audios, sampling_rate=SAMPLE_RATE,
                               return_tensors="pt", padding=True)
    return inputs.input_values, labels


test_loader_ft = DataLoader(range(len(test_audios)), batch_size=4,
                            shuffle=False, collate_fn=collate_audio_test)

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for input_values, labels in test_loader_ft:
        preds = model(input_values).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

acc_ft = accuracy_score(all_labels, all_preds)

best_space = "MFCC"
best_acc = max(results_mfcc.values())
if max(results_deep.values()) > best_acc:
    best_space = "Deep"
    best_acc = max(results_deep.values())
if acc_ft > best_acc:
    best_space = "Fine-tuned"
    best_acc = acc_ft

audios_all = train_audios + test_audios

if best_space == "MFCC":
    X_all = np.vstack([X_train_mfcc, X_test_mfcc])
    y_all = np.concatenate([y_train, y_test])
elif best_space == "Deep":
    X_all = np.vstack([X_train_deep, X_test_deep])
    y_all = np.concatenate([y_train_deep, y_test_deep])
else:
    model.eval()

    def extract_ft(audios):
        X = []
        for i in range(0, len(audios), 8):
            batch = audios[i:i + 8]
            inputs = feature_extractor(batch, sampling_rate=SAMPLE_RATE,
                                       return_tensors="pt", padding=True)
            with torch.no_grad():
                emb = model.encoder(inputs.input_values).last_hidden_state.mean(dim=1)
            X.append(emb.cpu().numpy())
        return np.vstack(X)

    X_all = np.vstack([extract_ft(train_audios), extract_ft(test_audios)])
    y_all = np.concatenate([train_labels, test_labels])

closest, farthest = find_extreme_pairs(X_all, y_all, n_pairs=2)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for i, (i1, i2, d) in enumerate(closest[:2]):
    plot_spectrogram(audios_all[i1], SAMPLE_RATE,
                     title=f"Closest pair {i+1}: idx {i1} (dist={d:.2f})",
                     ax=axes[0, i])
for i, (i1, i2, d) in enumerate(farthest[:2]):
    plot_spectrogram(audios_all[i1], SAMPLE_RATE,
                     title=f"Farthest pair {i+1}: idx {i1} (dist={d:.2f})",
                     ax=axes[1, i])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/spectrograms_extreme_pairs.png")
plt.show()
