"""
covariance.py — Online covariance estimator for Σ_t^{slv}

Σ_t^{slv} is the strategy-level covariance matrix used in the QP:

    b* = argmax_b [ ν^T b  −  (γ/2) b^T Σ b  −  λ_turn ‖b − b_{t-1}‖_1 ]

Because we have at most 3 strategies (A, B, C), Σ is a 3×3 matrix.
We update it each bar using an exponentially weighted moving average (EWMA)
over realised per-bar PnL of each engine.

Shrinkage
---------
We apply Ledoit-Wolf-style constant-correlation shrinkage:
  Σ_shrunk = (1 − ρ) * Σ_sample  +  ρ * Σ_target
where Σ_target = diag(σ_ii) * (ρ_avg * ones + (1−ρ_avg) * I)
and ρ_avg = average off-diagonal correlation.

This prevents near-singular Σ when strategies are highly correlated in
a short window.
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional
import numpy as np


class OnlineCovarianceEstimator:
    """
    Exponentially weighted covariance matrix over strategy PnL vectors.

    Parameters
    ----------
    engine_ids   : ordered list of engine IDs, e.g. ["A", "B", "C"]
    halflife     : EWMA half-life in bars
    min_obs      : minimum bars before Σ is considered valid
    shrink_alpha : Ledoit-Wolf shrinkage intensity [0, 1]
    floor_vol    : minimum diagonal volatility (prevents zero-variance blow-up)
    """

    def __init__(
        self,
        engine_ids: List[str],
        halflife: int = 40,
        min_obs: int = 20,
        shrink_alpha: float = 0.1,
        floor_vol: float = 0.001,
    ):
        self.engine_ids  = engine_ids
        self.n           = len(engine_ids)
        self.idx         = {e: i for i, e in enumerate(engine_ids)}
        self.halflife    = halflife
        self.min_obs     = min_obs
        self.shrink_alpha = shrink_alpha
        self.floor_vol   = floor_vol

        alpha = 1 - np.exp(-np.log(2) / halflife)
        self._alpha = alpha

        # EWMA state
        self._mean: np.ndarray = np.zeros(self.n)
        self._cov:  np.ndarray = np.eye(self.n) * (floor_vol ** 2)
        self._obs:  int = 0
        self._initialized = False

    def update(self, pnl_vector: Dict[str, float]):
        """
        Feed one bar's realized PnL for each engine.
        pnl_vector : {engine_id: realized_pnl_this_bar}
        """
        x = np.array([pnl_vector.get(e, 0.0) for e in self.engine_ids])

        if not self._initialized:
            self._mean = x.copy()
            self._initialized = True
        else:
            # EWMA mean update
            delta      = x - self._mean
            self._mean = self._mean + self._alpha * delta
            # EWMA covariance update (outer product of deviation)
            dev  = x - self._mean
            self._cov = (1 - self._alpha) * self._cov + self._alpha * np.outer(dev, dev)

        self._obs += 1

    def get(self) -> np.ndarray:
        """
        Return the current shrunk covariance matrix Σ_t^{slv}.
        Falls back to identity (scaled) if not enough observations.
        """
        if self._obs < self.min_obs:
            return np.eye(self.n) * (self.floor_vol ** 2)

        Sigma = self._cov.copy()

        # Enforce minimum diagonal (floor vol)
        for i in range(self.n):
            if Sigma[i, i] < self.floor_vol ** 2:
                Sigma[i, i] = self.floor_vol ** 2

        # Ledoit-Wolf constant-correlation shrinkage
        sigma_vec = np.sqrt(np.diag(Sigma))
        corr = Sigma / (np.outer(sigma_vec, sigma_vec) + 1e-12)
        avg_corr = (corr.sum() - self.n) / (self.n * (self.n - 1) + 1e-9)
        target_corr = avg_corr * np.ones((self.n, self.n)) + (1 - avg_corr) * np.eye(self.n)
        target_cov  = np.outer(sigma_vec, sigma_vec) * target_corr

        Sigma_shrunk = (1 - self.shrink_alpha) * Sigma + self.shrink_alpha * target_cov

        # Ensure positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(Sigma_shrunk)
        eigvals = np.clip(eigvals, self.floor_vol ** 2, None)
        Sigma_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return Sigma_psd
