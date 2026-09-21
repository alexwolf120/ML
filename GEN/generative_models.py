import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from imblearn.over_sampling import SMOTE

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

class GaussianImageModel:
    def __init__(self, n_components=1, latent=False):
        self.n_components = n_components
        self.latent = latent
        self.models = {}

    def _flatten(self, x):
        x = _to_numpy(x)
        return x.reshape(x.shape[0], -1)

    def fit(self, X, y):
        X = self._flatten(X)
        y = _to_numpy(y).astype(int)
        for c in np.unique(y):
            gm = GaussianMixture(
                n_components=self.n_components,
                covariance_type="full" if self.latent else "diag",
                random_state=0,
            )
            gm.fit(X[y == c])
            self.models[int(c)] = gm

    def sample(self, y, n_per_class=1):
        xs, ys = [], []
        for c in np.unique(y):
            gm = self.models[int(c)]
            s = gm.sample(n_per_class)[0]
            xs.append(s)
            ys.append(np.full(n_per_class, c))
        X = np.concatenate(xs, axis=0)
        Y = np.concatenate(ys, axis=0)
        return X, Y

class SMOTEModel:
    def __init__(self, k_neighbors=5, latent=False):
        self.k = k_neighbors
        self.latent = latent
        self.X_res = None
        self.y_res = None

    def fit_resample(self, X, y, target_per_class=None):
        X = _to_numpy(X)
        X = X.reshape(X.shape[0], -1)
        y = _to_numpy(y).astype(int)

        if target_per_class is not None:
            strategy = {}
            for c in np.unique(y):
                n_current = int((y == c).sum())
                n_target = max(target_per_class, n_current + 1)
                strategy[int(c)] = n_target
        else:
            strategy = "auto"

        sm = SMOTE(k_neighbors=self.k, sampling_strategy=strategy, random_state=0)
        self.X_res, self.y_res = sm.fit_resample(X, y)
        return self.X_res, self.y_res

    def sample_class(self, c, n):
        idx = np.where(self.y_res == c)[0]
        pick = np.random.choice(idx, size=n, replace=len(idx) < n)
        return self.X_res[pick]

def make_class_mixture(n_samples, num_classes, weights=None):
    if weights is None:
        weights = np.ones(num_classes) / num_classes
    y = np.random.choice(num_classes, size=n_samples, p=weights)
    return y