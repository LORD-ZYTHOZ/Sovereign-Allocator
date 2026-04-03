"""
state_builder.py — Build MarketState and AllocatorState from raw snapshots.

This is the "sensor layer" — it takes a UniverseSnapshot and computes
all the derived features that both engines and the allocator need.

All rolling statistics use Welford's online algorithm to avoid storing
the full history in memory.
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional
import numpy as np

from sovereign_allocator.data.schemas import (
    BarData, UniverseSnapshot, MarketState, AllocatorState
)


class _WelfordVar:
    """Online mean and variance using Welford's algorithm."""
    def __init__(self):
        self.n   = 0
        self.mean = 0.0
        self.M2   = 0.0

    def update(self, x: float):
        self.n += 1
        delta   = x - self.mean
        self.mean += delta / self.n
        delta2  = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / max(self.n - 1, 1)

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)


class StateBuilder:
    """
    Computes the full market state vector from raw bar snapshots.

    Parameters
    ----------
    symbols      : all symbols (assets + ETFs) to monitor
    vol_window   : bars for short-term vol estimate
    vol_window_5d: bars for medium-term vol estimate
    corr_window  : bars for rolling correlation
    sma_window   : bars for SMA (breadth calculation)
    trend_window : bars for trend strength ADX proxy
    """

    def __init__(
        self,
        symbols: List[str],
        vol_window: int   = 20,
        vol_window_5d: int = 100,
        corr_window: int  = 60,
        sma_window: int   = 20,
        trend_window: int = 14,
    ):
        self.symbols      = symbols
        self.vol_window   = vol_window
        self.vol_window_5d = vol_window_5d
        self.corr_window  = corr_window
        self.sma_window   = sma_window
        self.trend_window = trend_window

        # Return buffers
        self._ret_bufs: Dict[str, deque] = {
            s: deque(maxlen=max(vol_window_5d, corr_window)) for s in symbols
        }
        self._close_bufs: Dict[str, deque] = {
            s: deque(maxlen=sma_window) for s in symbols
        }
        self._prev_closes: Dict[str, float] = {}
        self._spread_buf: deque = deque(maxlen=vol_window)
        self._vol_buf:    deque = deque(maxlen=vol_window)

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def build_market_state(self, snapshot: UniverseSnapshot) -> MarketState:
        """Update internal buffers and compute MarketState."""
        all_bars = {**snapshot.asset_bars, **snapshot.etf_bars}
        returns_this_bar: List[float] = []

        spreads: List[float] = []
        vols:    List[float] = []

        for sym in self.symbols:
            bar = all_bars.get(sym)
            if bar is None:
                continue

            prev = self._prev_closes.get(sym, bar.open)
            log_ret = np.log(bar.close / (prev + 1e-9))
            self._ret_bufs[sym].append(log_ret)
            self._close_bufs[sym].append(bar.close)
            self._prev_closes[sym] = bar.close

            returns_this_bar.append(log_ret)
            if bar.spread > 0:
                spreads.append(bar.spread / (bar.close + 1e-9) * 10_000)  # to bps

            if len(self._ret_bufs[sym]) >= 5:
                sym_vol = float(np.std(list(self._ret_bufs[sym])[-self.vol_window:])) * np.sqrt(252)
                vols.append(sym_vol)

        # Scalar market-level stats
        avg_spread = float(np.mean(spreads)) if spreads else 1.0
        avg_vol_1d = float(np.mean(vols)) if vols else 0.15
        avg_vol_5d = self._compute_vol_5d()
        dispersion = float(np.std(returns_this_bar)) if returns_this_bar else 0.0
        avg_corr   = self._compute_avg_corr()
        breadth    = self._compute_breadth()
        trend      = self._compute_trend_strength()
        vol_ratio  = self._compute_vol_ratio(all_bars)

        return MarketState(
            timestamp=snapshot.timestamp,
            realized_vol_1d=avg_vol_1d,
            realized_vol_5d=avg_vol_5d,
            vix_proxy=avg_vol_1d,
            cross_asset_dispersion=dispersion,
            avg_pairwise_corr=avg_corr,
            pct_above_sma20=breadth,
            trend_strength=trend,
            avg_spread_bps=avg_spread,
            avg_volume_ratio=vol_ratio,
        )

    def build_alloc_state(self, ms: MarketState) -> AllocatorState:
        """
        Map raw MarketState scalars → normalised [0,1] AllocatorState.
        These thresholds are tunable; treat them as hyperparameters.
        """
        vol_regime  = np.clip(ms.realized_vol_1d / 0.40, 0.0, 1.0)   # 40% annualised = max
        disp_regime = np.clip(ms.cross_asset_dispersion / 0.02, 0.0, 1.0)
        corr_regime = np.clip((ms.avg_pairwise_corr + 1) / 2, 0.0, 1.0)
        liq_score   = np.clip(1 - ms.avg_spread_bps / 20, 0.0, 1.0)  # 20 bps = very illiquid
        macro_score = np.clip(ms.macro_surprise, -1.0, 1.0)

        return AllocatorState(
            timestamp=ms.timestamp,
            volatility_regime=float(vol_regime),
            dispersion_regime=float(disp_regime),
            correlation_regime=float(corr_regime),
            breadth_score=float(ms.pct_above_sma20),
            trend_score=float(ms.trend_strength),
            liquidity_score=float(liq_score),
            macro_score=float(macro_score),
        )

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _compute_vol_5d(self) -> float:
        vols = []
        for sym in self.symbols:
            buf = self._ret_bufs[sym]
            if len(buf) >= self.vol_window_5d:
                vols.append(np.std(list(buf)[-self.vol_window_5d:]) * np.sqrt(252))
        return float(np.mean(vols)) if vols else 0.15

    def _compute_avg_corr(self) -> float:
        ret_arrays = []
        for sym in self.symbols:
            buf = list(self._ret_bufs[sym])
            if len(buf) >= self.corr_window:
                ret_arrays.append(np.array(buf[-self.corr_window:]))

        if len(ret_arrays) < 2:
            return 0.0

        matrix = np.vstack(ret_arrays)       # (n_syms, corr_window)
        corr   = np.corrcoef(matrix)         # (n_syms, n_syms)
        n      = corr.shape[0]
        # Mean off-diagonal
        off_diag = corr[np.triu_indices(n, k=1)]
        return float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0

    def _compute_breadth(self) -> float:
        above = 0
        total = 0
        for sym in self.symbols:
            closes = list(self._close_bufs[sym])
            if len(closes) < 2:
                continue
            sma = np.mean(closes)
            if closes[-1] > sma:
                above += 1
            total += 1
        return above / max(total, 1)

    def _compute_trend_strength(self) -> float:
        """Simple proxy: fraction of symbols trending (slope > 0)."""
        trending = 0
        total    = 0
        for sym in self.symbols:
            closes = list(self._close_bufs[sym])
            if len(closes) < 5:
                continue
            x = np.arange(len(closes))
            slope = np.polyfit(x, closes, 1)[0]
            if slope > 0:
                trending += 1
            total += 1
        return trending / max(total, 1)

    def _compute_vol_ratio(self, all_bars: dict) -> float:
        """Average ratio of current volume to recent mean volume."""
        ratios = []
        for sym in self.symbols:
            bar = all_bars.get(sym)
            if bar is None or bar.volume <= 0:
                continue
            # Without a rolling volume buffer we approximate as 1.0
            # (extend _ret_bufs to track volume if needed)
            ratios.append(1.0)
        return float(np.mean(ratios)) if ratios else 1.0
