import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from data_utils import (
    soft_argmax_cross_entropy,
    soft_argmax_cross_entropy_backward,
)


FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)


class Model:
    def __init__(self, layers, name, short=None):
        self.layers = layers
        self.name = name
        self.short = short if short is not None else name

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self):
        for layer in self.layers:
            layer.update()

    def params_count(self):
        return sum(layer.params_count() for layer in self.layers)

    def depth(self):
        return len(self.layers)

    def get_name(self):
        return self.name

    def get_short(self):
        return self.short


def _train_step(model, x, y):
    logits = model.forward(x)
    loss = soft_argmax_cross_entropy(logits, y)
    grad = soft_argmax_cross_entropy_backward(logits, y)
    model.backward(grad)
    model.update()
    return loss


def _accuracy(model, x, y):
    return accuracy_score(y, np.argmax(model.forward(x), axis=1))


def run_model(model, x_train, x_test, y_train, y_test,
              epochs=200, batch_size=32, eval_every=5, seed=0):
    rng = np.random.default_rng(seed)
    n = x_train.shape[0]

    train_losses, test_acc = [], []

    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            epoch_loss += _train_step(model, x_train[idx], y_train[idx]) * len(idx)
        epoch_loss /= n

        if epoch % eval_every == 0 or epoch == epochs - 1:
            train_losses.append(epoch_loss)
            test_acc.append(_accuracy(model, x_test, y_test))

    return {
        "name": model.get_name(),
        "short": model.get_short(),
        "loss": train_losses,
        "acc": test_acc,
        "final_acc": test_acc[-1],
        "params": model.params_count(),
        "depth": model.depth(),
    }


def _smooth(y, k=3):
    if len(y) < k or k <= 1:
        return np.asarray(y, dtype=float)
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="valid")


def _grouped_curve(results, key, round_to=None):
    grouped = {}
    for r in results:
        v = r[key]
        if round_to:
            v = int(round(v / round_to)) * round_to
        grouped.setdefault(v, []).append(r["final_acc"])
    xs = sorted(grouped)
    ys = [float(np.mean(grouped[k])) for k in xs]
    return xs, ys


def plot_learning_curves(results, title, file_name=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(title, fontsize=15)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    ax = axes[0]
    for r, c in zip(results, colors):
        ax.plot(_smooth(r["loss"], 3),
                label=r["short"], color=c, lw=1.6)
    ax.set_xlabel("evaluation index")
    ax.set_ylabel("loss")
    ax.set_title("Learning curves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              title="Model", title_fontsize=9)

    ax = axes[1]
    for r, c in zip(results, colors):
        ax.plot(_smooth(r["acc"], 3),
                label=r["short"], color=c, lw=1.6)
    ax.set_xlabel("evaluation index")
    ax.set_ylabel("accuracy")
    ax.set_title("Test accuracy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9,
              title="Model", title_fontsize=9)

    plt.tight_layout()
    if file_name:
        plt.savefig(os.path.join(FIG_DIR, file_name), dpi=130,
                    bbox_inches="tight")
    plt.close()


def plot_capacity_curves(results, title, file_name=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(title, fontsize=15)

    ax = axes[0]
    xs, ys = _grouped_curve(results, "params", round_to=100)
    ax.plot(xs, ys, "o-", color="green", lw=1.6, ms=8)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), fontsize=9,
                    xytext=(6, 6), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("number of parameters")
    ax.set_ylabel("mean accuracy")
    ax.set_title("Accuracy vs number of parameters")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    xs, ys = _grouped_curve(results, "depth")
    ax.plot(xs, ys, "o-", color="green", lw=1.6, ms=8)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), fontsize=9,
                    xytext=(6, 6), textcoords="offset points")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("number of layers")
    ax.set_ylabel("mean accuracy")
    ax.set_title("Accuracy vs number of layers")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if file_name:
        plt.savefig(os.path.join(FIG_DIR, file_name), dpi=130,
                    bbox_inches="tight")
    plt.close()


def plot_results(results, title, file_name_prefix):
    plot_learning_curves(results, title, f"{file_name_prefix}_curves.png")
    plot_capacity_curves(results, title, f"{file_name_prefix}_capacity.png")