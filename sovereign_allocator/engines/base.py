"""
base.py — Abstract base class all strategy engines must implement.

Each engine:
  1. Receives a UniverseSnapshot + MarketState at every bar.
  2. Updates its internal model / running statistics.
  3. Returns a StrategySignal with its utility estimate.

Engines are stateful — they maintain rolling windows, online covariance
estimates, and PnL history internally. The allocator never touches internals.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List, Optional
import numpy as np

from sovereign_allocator.data.schemas import (
    UniverseSnapshot, MarketState, AllocatorState, StrategySignal
)


class BaseEngine(ABC):
    """
    Subclass and implement `compute_signal`.

    Parameters
    ----------
    engine_id    : "A" | "B" | "C"  (must be unique)
    lambda_u     : penalty weight on risk score U_t^s
    lambda_c     : penalty weight on cost estimate Ĉ_t^s
    pnl_window   : bars of realized PnL history to keep for η̂ estimation
    """

    def __init__(
        self,
        engine_id: str,
        lambda_u: float = 1.0,
        lambda_c: float = 0.5,
        pnl_window: int = 60,
    ):
        self.engine_id  = engine_id
        self.lambda_u   = lambda_u
        self.lambda_c   = lambda_c
        self.pnl_window = pnl_window

        # Rolling realized PnL — used to estimate η̂ via EWMA
        self._pnl_history: Deque[float] = deque(maxlen=pnl_window)
        self._last_positions: dict = {}

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def step(
        self,
        snapshot: UniverseSnapshot,
        state: MarketState,
        alloc_state: Optional["AllocatorState"] = None,
    ) -> StrategySignal:
        """
        Called once per bar by the main loop.
        Computes signal, records PnL, returns a StrategySignal.
        """
        signal = self.compute_signal(snapshot, state, alloc_state)

        # Back-fill realized PnL from last period's positions
        if self._last_positions:
            realized = self._calc_realized_pnl(snapshot)
            self._pnl_history.append(realized)
            signal.eta_hat = self._estimate_eta()

        # Build utility score:  ν = η̂ − λ_U * U − λ_C * Ĉ
        signal.utility_raw = (
            signal.eta_hat
            - self.lambda_u * signal.risk_score
            - self.lambda_c * signal.cost_estimate
        )

        self._last_positions = signal.positions.copy()
        return signal

    # ──────────────────────────────────────────
    # Must override
    # ──────────────────────────────────────────

    @abstractmethod
    def compute_signal(
        self,
        snapshot: UniverseSnapshot,
        state: MarketState,
        alloc_state: Optional["AllocatorState"] = None,
    ) -> StrategySignal:
        """
        Produce the raw signal for this bar.
        Set positions, risk_score, cost_estimate, confidence, metadata.
        eta_hat will be overwritten by the base class using realized PnL history,
        but you can set an initial model-based estimate here as a prior.
        """
        ...

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _calc_realized_pnl(self, snapshot: UniverseSnapshot) -> float:
        """
        Simple bar-over-bar PnL: sum of (position * return) for each held asset.
        Override for more accurate fill simulation.
        """
        pnl = 0.0
        for symbol, weight in self._last_positions.items():
            bar = snapshot.asset_bars.get(symbol)
            if bar is None:
                continue
            ret = (bar.close - bar.open) / (bar.open + 1e-9)
            pnl += weight * ret
        return pnl

    def _estimate_eta(self, half_life: int = 20) -> float:
        """
        EWMA estimate of expected edge η̂ from recent realized PnL history.
        half_life controls decay speed (in bars).
        """
        if not self._pnl_history:
            return 0.0
        arr = np.array(self._pnl_history)
        alpha = 1 - np.exp(-np.log(2) / max(half_life, 1))
        weights = (1 - alpha) ** np.arange(len(arr) - 1, -1, -1)
        weights /= weights.sum()
        ewma_return = float(np.dot(weights, arr))
        # Annualise (assume ~252 trading bars per year; adjust for your timeframe)
        return ewma_return * 252
