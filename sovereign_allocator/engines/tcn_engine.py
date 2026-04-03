"""
Engine A — Causal Temporal Convolutional Network (TCN)
=======================================================

What it does
------------
Uses a causal (no look-ahead) dilated convolutional network to model the
conditional distribution of next-bar returns for each asset.

  P(r_{t+1} | r_{t}, r_{t-1}, ..., r_{t-L})

Outputs a next-step probability estimate (positive / flat / negative) and
converts that into a directional position for each symbol.

Architecture sketch
-------------------
Input:  (batch=1, channels=features, length=receptive_field)
        Features per bar: [return, log_volume, spread_bps, vol_ratio]

Stack of N residual TCN blocks, each block:
  dilated causal conv → weight-norm → GELU → dropout
  → 1x1 conv to match channels → residual add

Output head: linear → softmax over {short, flat, long}

The full PyTorch model is in configs/tcn_model.py (or bring your own).
This file is the *engine wrapper* that feeds data, runs inference,
and converts logits → positions → StrategySignal.

NOTE: for backtesting without a trained model, set USE_STUB=True in the
config and the engine will use a simple momentum proxy instead.
"""

from __future__ import annotations
from collections import deque
from typing import Dict, Optional
import numpy as np

from sovereign_allocator.data.schemas import (
    UniverseSnapshot, MarketState, AllocatorState, StrategySignal
)
from sovereign_allocator.engines.base import BaseEngine


# ──────────────────────────────────────────────
# Lightweight causal TCN (pure NumPy stub)
# Replace with your PyTorch / MLX version.
# ──────────────────────────────────────────────

class _CausalTCNStub:
    """
    Momentum proxy that mimics what a trained TCN would produce.
    Returns P(long), P(flat), P(short) given a return series.
    """

    def __init__(self, lookback: int = 30):
        self.lookback = lookback

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """
        returns : 1-D array of recent log-returns (newest last)
        Returns : array([p_short, p_flat, p_long])
        """
        if len(returns) < 5:
            return np.array([1/3, 1/3, 1/3])

        recent = returns[-self.lookback:]
        mu    = np.mean(recent)
        sigma = np.std(recent) + 1e-8
        z     = mu / sigma   # Sharpe-like z-score

        # Soft probabilities via a simple sigmoid squeeze
        p_long  = 1 / (1 + np.exp(-3 * z))
        p_short = 1 / (1 + np.exp( 3 * z))
        p_flat  = 1 - p_long - p_short
        p_flat  = max(p_flat, 0.05)

        total = p_long + p_flat + p_short
        return np.array([p_short, p_flat, p_long]) / total


# ──────────────────────────────────────────────
# Engine A
# ──────────────────────────────────────────────

class TCNEngine(BaseEngine):
    """
    Causal TCN engine.

    Parameters
    ----------
    symbols        : list of asset symbols this engine trades
    lookback       : bars of history to feed the TCN
    model          : optional pre-trained TCN; if None, uses stub
    position_cap   : max absolute weight per symbol
    lambda_u / lambda_c : inherited risk / cost penalty weights
    """

    ENGINE_ID = "A"

    def __init__(
        self,
        symbols: list,
        lookback: int = 60,
        model=None,
        position_cap: float = 0.25,
        lambda_u: float = 1.0,
        lambda_c: float = 0.5,
    ):
        super().__init__(
            engine_id=self.ENGINE_ID,
            lambda_u=lambda_u,
            lambda_c=lambda_c,
        )
        self.symbols      = symbols
        self.lookback     = lookback
        self.position_cap = position_cap
        self.model = model or _CausalTCNStub(lookback=lookback)

        # Rolling return buffers per symbol
        self._return_buffers: Dict[str, deque] = {
            s: deque(maxlen=lookback + 1) for s in symbols
        }
        self._prev_closes: Dict[str, float] = {}

    # ──────────────────────────────────────────
    # Core signal logic
    # ──────────────────────────────────────────

    def compute_signal(
        self,
        snapshot: UniverseSnapshot,
        state: MarketState,
        alloc_state: Optional[AllocatorState] = None,
    ) -> StrategySignal:

        positions: Dict[str, float] = {}
        confidence_scores = []

        for symbol in self.symbols:
            bar = snapshot.asset_bars.get(symbol)
            if bar is None:
                continue

            # Update return buffer
            prev = self._prev_closes.get(symbol, bar.open)
            log_ret = np.log(bar.close / (prev + 1e-9))
            self._return_buffers[symbol].append(log_ret)
            self._prev_closes[symbol] = bar.close

            if len(self._return_buffers[symbol]) < 10:
                continue

            returns = np.array(self._return_buffers[symbol])
            probs   = self.model.predict_proba(returns)   # [p_short, p_flat, p_long]

            p_short, p_flat, p_long = probs
            # Directional score: positive → long, negative → short
            direction_score = p_long - p_short
            confidence      = abs(direction_score)

            # Scale to position [-cap, +cap]
            raw_pos = direction_score * self.position_cap
            positions[symbol] = float(np.clip(raw_pos, -self.position_cap, self.position_cap))
            confidence_scores.append(confidence)

        # Risk score: volatility-adjusted drawdown risk
        vol = state.realized_vol_1d
        vol_regime = alloc_state.volatility_regime if alloc_state is not None else 0.5
        risk_score = vol * (1 + vol_regime)   # higher in stressed regimes

        # Cost estimate: proportional to spread + position churn
        cost_est = state.avg_spread_bps / 10_000          # convert bps → fraction

        avg_conf = float(np.mean(confidence_scores)) if confidence_scores else 0.0

        return StrategySignal(
            engine_id=self.ENGINE_ID,
            timestamp=snapshot.timestamp,
            eta_hat=0.0,              # overwritten by base class EWMA
            utility_raw=0.0,          # overwritten by base class
            risk_score=risk_score,
            cost_estimate=cost_est,
            positions=positions,
            confidence=avg_conf,
            metadata={
                "model_type": "CausalTCN",
                "n_active_symbols": len(positions),
            },
        )
