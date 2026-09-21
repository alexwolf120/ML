import numpy as np
from data_utils import adaptive_gd

def identity(x):
    return x

def identity_grad(x):
    return np.ones_like(x)

def relu(x):
    return np.maximum(x, 0)

def relu_grad(x):
    return (x > 0).astype(np.float32)

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - np.tanh(x) ** 2

ACTIVATIONS = {
    "identity": (identity, identity_grad),
    "relu": (relu, relu_grad),
    "tanh": (tanh, tanh_grad),
}

class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def update(self):
        pass

    def params_count(self):
        return 0

class LinearLayer(Layer):

    def __init__(self, in_size, out_size, activation="identity"):
        self.A = np.random.randn(in_size, out_size) / 10
        self.b = np.zeros(out_size)
        self.act, self.act_grad = ACTIVATIONS[activation]
        self.histories = [None, None]

    def forward(self, x):
        self.x = x
        self.z = x @ self.A + self.b
        return self.act(self.z)

    def backward(self, grad):
        dz = grad * self.act_grad(self.z)
        self.A_grad = self.x.T @ dz
        self.b_grad = dz.sum(axis=0)
        return dz @ self.A.T

    def update(self):
        self.A, self.b = adaptive_gd(
            [self.A, self.b],
            [self.A_grad, self.b_grad],
            self.histories,
        )

    def params_count(self):
        return self.A.size + self.b.size

class RBFLayer(Layer):

    def __init__(self, in_size, out_size, l=1.0):
        self.A = np.random.randn(in_size, out_size) * 0.5
        self.l = float(l)
        self.histories = [None]

    def forward(self, x):
        self.x = x
        diff = x[:, None, :] - self.A.T[None, :, :]
        dist_sq = np.einsum("njd,njd->nj", diff, diff)
        self.diff = diff
        self.y = np.exp(-self.l * dist_sq)
        return self.y

    def backward(self, grad):
        weighted = grad[:, :, None] * (-2.0 * self.l) \
                   * self.y[:, :, None] * self.diff
        self.A_grad = -weighted.sum(axis=0).T
        return weighted.sum(axis=1)

    def update(self):
        (self.A,) = adaptive_gd([self.A], [self.A_grad], self.histories)

    def params_count(self):
        return self.A.size

class Residual(Layer):

    def __init__(self, layer):
        self.layer = layer

    def forward(self, x):
        y = self.layer.forward(x)
        if y.shape != x.shape:
            raise ValueError(
                f"Residual требует совпадения форм, получено {x.shape} и {y.shape}"
            )
        return x + y

    def backward(self, grad):
        return grad + self.layer.backward(grad)

    def update(self):
        self.layer.update()

    def params_count(self):
        return self.layer.params_count()