import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SELECTED_COLUMNS = [
    "price_rub",
    "cpu_cores",
    "storage_gb",
    "screen_diag_inch",
    "ram_type_DDR4",
    "ram_type_DDR5",
    "ram_type_LPDDR5",
    "os_Windows 11 Home",
    "os_Windows 11 Pro",
    "os_macOS",
]

TARGET_COLUMN = "ram_gb"


def load_data(path="data.csv", test_size=0.2, random_state=42):

    data = pd.read_csv(path, sep=',', decimal='.')

    missing = [c for c in SELECTED_COLUMNS + [TARGET_COLUMN]
               if c not in data.columns]
    if missing:
        raise ValueError(f"В data.csv нет колонок: {missing}")

    ram = data[TARGET_COLUMN].values.astype(np.float32)
    q33, q66 = np.quantile(ram, [0.33, 0.66])

    y_raw = np.zeros_like(ram, dtype=np.int32)
    y_raw[ram >= q33] = 1
    y_raw[ram >= q66] = 2

    x = data[SELECTED_COLUMNS].values.astype(np.float32)

    if np.isnan(x).any():
        col_mean = np.nanmean(x, axis=0)
        idx = np.where(np.isnan(x))
        x[idx] = np.take(col_mean, idx[1])

    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_raw, test_size=test_size, random_state=random_state,
        stratify=y_raw,
    )
    return x_train, x_test, y_train, y_test

def log_softmax(x):
    x_max = np.max(x, axis=1, keepdims=True)
    x_shifted = x - x_max
    return x_shifted - np.log(np.sum(np.exp(x_shifted), axis=1, keepdims=True) + 1e-12)

def soft_argmax_cross_entropy(x, targets):
    n = x.shape[0]
    one_hot = np.zeros_like(x)
    one_hot[np.arange(n), targets] = 1
    return -np.sum(one_hot * log_softmax(x)) / n

def soft_argmax_cross_entropy_backward(x, targets):
    n = x.shape[0]
    one_hot = np.zeros_like(x)
    one_hot[np.arange(n), targets] = 1
    return (np.exp(log_softmax(x)) - one_hot) / n

class AdamState:
    __slots__ = ("m", "v", "t")

    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_LR = 0.0005
_ADAM_EPS = 1e-8


def adaptive_gd(params, grads, histories, learning_rate=_ADAM_LR):
    new_params = []
    for i, (p, g) in enumerate(zip(params, grads)):
        h = histories[i]
        if not isinstance(h, AdamState):
            h = AdamState(g.shape)
            histories[i] = h

        h.t += 1
        h.m = _ADAM_BETA1 * h.m + (1.0 - _ADAM_BETA1) * g
        h.v = _ADAM_BETA2 * h.v + (1.0 - _ADAM_BETA2) * (g * g)

        m_hat = h.m / (1.0 - _ADAM_BETA1 ** h.t)
        v_hat = h.v / (1.0 - _ADAM_BETA2 ** h.t)

        step = learning_rate * m_hat / (np.sqrt(v_hat) + _ADAM_EPS)
        new_params.append(p - step)
    return new_params