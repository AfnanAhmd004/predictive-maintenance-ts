"""Anomaly detectors trained on healthy data only.

All detectors share `fit(X_healthy)` and `score(X) -> anomaly score`, and pick
an alarm threshold from a high quantile of healthy-data scores.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from torch import nn


class _Base:
    quantile = 0.995

    def calibrate(self, X):
        self.threshold_ = float(np.quantile(self.score(X), self.quantile))
        return self

    def alarms(self, X):
        return self.score(X) > self.threshold_


class ZScore(_Base):
    """Max absolute z-score across sensors."""

    def fit(self, X):
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        return self.calibrate(X)

    def score(self, X):
        return np.abs((X - self.mu_) / self.sd_).max(1)


class PCAResidual(_Base):
    """Squared prediction error (SPE / Q-statistic) outside the healthy principal subspace.

    Catches faults that break the normal correlation between sensors even when
    each sensor stays inside its own normal range.
    """

    def __init__(self, n_components: int = 2):
        self.k = n_components

    def fit(self, X):
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu_) / self.sd_
        _, _, vt = np.linalg.svd(Z, full_matrices=False)
        self.P_ = vt[: self.k].T
        return self.calibrate(X)

    def score(self, X):
        Z = (X - self.mu_) / self.sd_
        resid = Z - Z @ self.P_ @ self.P_.T
        return (resid**2).sum(1)


class IForest(_Base):
    def __init__(self, seed: int = 0):
        self.model = IsolationForest(n_estimators=200, random_state=seed)

    def fit(self, X):
        self.model.fit(X)
        return self.calibrate(X)

    def score(self, X):
        return -self.model.score_samples(X)


class WindowAutoencoder(_Base):
    """Dense autoencoder over short multi-sensor windows; score = reconstruction error."""

    def __init__(self, window: int = 12, latent: int = 6, epochs: int = 40, seed: int = 0):
        self.window, self.latent, self.epochs, self.seed = window, latent, epochs, seed

    def _windows(self, Z):
        w = self.window
        out = np.stack([Z[i : i + w].ravel() for i in range(len(Z) - w + 1)]).astype(np.float32)
        return torch.from_numpy(out)

    def fit(self, X):
        torch.manual_seed(self.seed)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        W = self._windows((X - self.mu_) / self.sd_)
        d = W.shape[1]
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, self.latent), nn.ReLU(),
                                 nn.Linear(self.latent, 32), nn.ReLU(), nn.Linear(32, d))
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        for _ in range(self.epochs):
            for i in torch.randperm(len(W)).split(128):
                opt.zero_grad()
                loss = ((self.net(W[i]) - W[i]) ** 2).mean()
                loss.backward()
                opt.step()
        return self.calibrate(X)

    def score(self, X):
        W = self._windows((X - self.mu_) / self.sd_)
        with torch.no_grad():
            err = ((self.net(W) - W) ** 2).mean(1).numpy()
        return np.r_[np.full(self.window - 1, err[0]), err]  # align score with the window's last hour
