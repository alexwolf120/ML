import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import struct
import os
import gzip


def _open_idx(filename):
    if filename.endswith('.gz'):
        return gzip.open(filename, 'rb')
    return open(filename, 'rb')


def read_idx_images(filename):
    with _open_idx(filename) as f:
        fmt, count, height, width = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(count, height, width).astype(np.float32)
    return images


def read_idx_labels(filename):
    with _open_idx(filename) as f:
        struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def _find_file(data_dir, candidates, description):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Папка '{data_dir}' не существует (ожидалась для {description})."
        )
    for name in candidates:
        for candidate in (name, name + '.gz'):
            path = os.path.join(data_dir, candidate)
            if os.path.exists(path):
                return path

    files = os.listdir(data_dir)
    raise FileNotFoundError(
        f"Не найден файл для {description} в папке '{data_dir}'.\n"
        f"Искали: {candidates} (и с .gz)\n"
        f"В папке лежат: {files}"
    )


def read_idx(data_dir, prefix=''):
    train_images_path = _find_file(data_dir, [
        f'{prefix}train-images-idx3-ubyte',
        f'{prefix}train-images.idx3-ubyte',
    ], 'train images')
    train_labels_path = _find_file(data_dir, [
        f'{prefix}train-labels-idx1-ubyte',
        f'{prefix}train-labels.idx1-ubyte',
    ], 'train labels')
    test_images_path = _find_file(data_dir, [
        f'{prefix}t10k-images-idx3-ubyte',
        f'{prefix}t10k-images.idx3-ubyte',
    ], 'test images')
    test_labels_path = _find_file(data_dir, [
        f'{prefix}t10k-labels-idx1-ubyte',
        f'{prefix}t10k-labels.idx1-ubyte',
    ], 'test labels')

    train_images = read_idx_images(train_images_path) / 255.0
    train_labels = read_idx_labels(train_labels_path)
    test_images = read_idx_images(test_images_path) / 255.0
    test_labels = read_idx_labels(test_labels_path)

    train_dataset = TensorDataset(
        torch.FloatTensor(train_images.copy()).unsqueeze(1),
        torch.LongTensor(train_labels.copy())
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_images.copy()).unsqueeze(1),
        torch.LongTensor(test_labels.copy())
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


def load_mnist(data_dir='data/Mnist'):
    return read_idx(data_dir, prefix='')


def load_fashion_mnist(data_dir='data/FashionMnist'):
    return read_idx(data_dir, prefix='')
