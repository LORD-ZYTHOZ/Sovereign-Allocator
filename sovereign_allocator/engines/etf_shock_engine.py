"""
Engine C — ETF Shock Propagation Strategy (macro / top-down)
=============================================================

What it does
------------
Treats large ETF moves as macro "levers" that have predictable downstream
effects on the underlying assets or related instruments.

Pipeline
--------
1.  Shock detection
    For each ETF at time t, compute a z-scored return:

        z_t^{ETF} = (r_t^{ETF} − μ_r) / σ_r

    where μ_r, σ_r are the rolling mean and std over the past `shock_window` bars.

    A shock is declared when |z_t^{ETF}| > shock_threshold (e.g. 2.0).

2.  Shock propagation
    Each asset j has a pre-estimated loading (beta) on each ETF:

        β_{j, ETF}  ≈  Cov(r^j, r^{ETF}) / Var(r^{ETF})

    The expected conditional drift injected into asset j over the next
    `signal_decay` bars is:

        drift_{t,j} = Σ_{ETF} β_{j,ETF} × (shock_amplitude × sign(z_t^{ETF}))

    Positive drift → expected upward bias → go long.

3.  Signal decay
    The injected drift decays linearly over `signal_decay` bars so that
    the position sizing tapers as the shock ages.

4.  Beta estimation
    Betas are re-estimated online every `beta_update_freq` bars using
    rolling OLS from the return buffers.

This engine captures the MACRO edge:
  - Top-down flow effects (e.g. large SPY inflow lifts tech names)
  - Sector ETF shocks propagate to single names
  - Cross-asset macro contagion (e.g. HYG dump → high-beta equity unwind)
"""

from __future__ import annotations
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import numpy as np

from sovereign_allocator.data.schemas import (
    UniverseSnapshot, MarketState, AllocatorState, StrategySignal
)
from sovereign_allocator.engines.base import BaseEngine


class ETFShockEngine(BaseEngine):
    """
    Parameters
    ----------
    asset_symbols      : single-name / futures to trade
    etf_symbols        : ETFs used as macro signals
    shock_window       : rolling window for z-score computation (bars)
    shock_threshold    : |z| threshold to declare a shock
    beta_window        : bars used to estimate ETF→asset betas
    beta_update_freq   : rebuild betas every N bars
    signal_decay       : bars over which injected drift linearly decays
    position_cap       : max abs weight per asset
    """

    ENGINE_ID = "C"

    def __init__(
        self,
        asset_symbols: List[str],
        etf_symbols: List[str],
        shock_window: int = 40,
        shock_threshold: float = 2.0,
        beta_window: int = 120,
        beta_update_freq: int = 30,
        signal_decay: int = 5,
        position_cap: float = 0.20,
        lambda_u: float = 1.0,
        lambda_c: float = 0.6,
    ):
        super().__init__(engine_id=self.ENGINE_ID, lambda_u=lambda_u, lambda_c=lambda_c)
        self.asset_symbols    = asset_symbols
        self.etf_symbols      = etf_symbols
        self.shock_window     = shock_window
        self.shock_threshold  = shock_threshold
        self.beta_window      = beta_window
        self.beta_update_freq = beta_update_freq
        self.signal_decay     = signal_decay
        self.position_cap     = position_cap

        # Rolling return buffers
        self._etf_returns:   Dict[str, deque] = {e: deque(maxlen=beta_window) for e in etf_symbols}
        self._asset_returns: Dict[str, deque] = {a: deque(maxlen=beta_window) for a in asset_symbols}
        self._prev_etf_closes:   Dict[str, float] = {}
        self._prev_asset_closes: Dict[str, float] = {}

        # Beta matrix: asset → {etf → beta}
        self._betas: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Active shock signals: {asset: (remaining_bars, drift_per_bar)}
        self._active_signals: Dict[str, Tuple[int, float]] = {}

        self._bar_count = 0

    # ──────────────────────────────────────────
    # Beta estimation
    # ──────────────────────────────────────────

    def _update_betas(self):
        """
        Re-estimate β_{asset, ETF} for every (asset, ETF) pair using
        rolling OLS: β = Cov(r^asset, r^ETF) / Var(r^ETF).
        """
        for etf in self.etf_symbols:
            etf_buf = np.array(self._etf_returns[etf])
            if len(etf_buf) < 20:
                continue
            var_etf = np.var(etf_buf) + 1e-10

            for asset in self.asset_symbols:
                asset_buf = np.array(self._asset_returns[asset])
                min_len = min(len(etf_buf), len(asset_buf))
                if min_len < 20:
                    continue
                cov = np.cov(asset_buf[-min_len:], etf_buf[-min_len:])[0, 1]
                self._betas[asset][etf] = cov / var_etf

    # ──────────────────────────────────────────
    # Shock detection
    # ──────────────────────────────────────────

    def _detect_shocks(self) -> Dict[str, float]:
        """
        Returns {etf: shock_z} for ETFs whose |z| > threshold this bar.
        """
        shocks = {}
        for etf in self.etf_symbols:
            buf = np.array(self._etf_returns[etf])
            if len(buf) < self.shock_window:
                continue
            recent = buf[-self.shock_window:]
            mu     = np.mean(recent[:-1])   # exclude current bar from mean
            sigma  = np.std(recent[:-1]) + 1e-8
            z      = (buf[-1] - mu) / sigma
            if abs(z) >= self.shock_threshold:
                shocks[etf] = float(z)
        return shocks

    # ──────────────────────────────────────────
    # Core signal logic
    # ──────────────────────────────────────────

    def compute_signal(
        self,
        snapshot: UniverseSnapshot,
        state: MarketState,
        alloc_state: Optional[AllocatorState] = None,
    ) -> StrategySignal:

        # 1. Update ETF return buffers
        for etf in self.etf_symbols:
            bar = snapshot.etf_bars.get(etf)
            if bar is None:
                continue
            prev = self._prev_etf_closes.get(etf, bar.open)
            self._etf_returns[etf].append(np.log(bar.close / (prev + 1e-9)))
            self._prev_etf_closes[etf] = bar.close

        # 2. Update asset return buffers
        for asset in self.asset_symbols:
            bar = snapshot.asset_bars.get(asset)
            if bar is None:
                continue
            prev = self._prev_asset_closes.get(asset, bar.open)
            self._asset_returns[asset].append(np.log(bar.close / (prev + 1e-9)))
            self._prev_asset_closes[asset] = bar.close

        self._bar_count += 1

        # 3. Periodically refresh betas
        if self._bar_count % self.beta_update_freq == 0:
            self._update_betas()

        # 4. Detect shocks this bar
        shocks = self._detect_shocks()

        # 5. If shocks found, inject new signals (override any decaying ones)
        if shocks:
            for asset in self.asset_symbols:
                total_drift = 0.0
                for etf, z in shocks.items():
                    beta = self._betas.get(asset, {}).get(etf, 0.0)
                    # Drift proportional to beta × shock_amplitude
                    shock_amp = np.sign(z) * min(abs(z) / 4.0, 1.0)   # cap at ±1
                    total_drift += beta * shock_amp

                if abs(total_drift) > 1e-6:
                    drift_per_bar = total_drift / self.signal_decay
                    self._active_signals[asset] = (self.signal_decay, drift_per_bar)

        # 6. Build positions from active (decaying) signals
        positions: Dict[str, float] = {}
        for asset, (bars_left, drift_pb) in list(self._active_signals.items()):
            if bars_left <= 0:
                del self._active_signals[asset]
                continue
            # Taper: linear decay, e.g. 5 bars: weights [1.0, 0.8, 0.6, 0.4, 0.2]
            taper = bars_left / self.signal_decay
            pos   = drift_pb * taper * self.signal_decay * self.position_cap
            positions[asset] = float(np.clip(pos, -self.position_cap, self.position_cap))
            self._active_signals[asset] = (bars_left - 1, drift_pb)

        # 7. Risk and cost
        # Macro shocks increase volatility exposure
        n_shocks   = len(shocks)
        risk_score = state.realized_vol_1d * (1 + 0.3 * n_shocks)
        cost_est   = state.avg_spread_bps / 10_000 * (1 + 0.1 * n_shocks)

        confidence = min(n_shocks / max(len(self.etf_symbols), 1), 1.0)

        return StrategySignal(
            engine_id=self.ENGINE_ID,
            timestamp=snapshot.timestamp,
            eta_hat=0.0,
            utility_raw=0.0,
            risk_score=risk_score,
            cost_estimate=cost_est,
            positions=positions,
            confidence=confidence,
            metadata={
                "shocks_detected": shocks,
                "n_active_signals": len(positions),
                "bar_count": self._bar_count,
            },
        )
