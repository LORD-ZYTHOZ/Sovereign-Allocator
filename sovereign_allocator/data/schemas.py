"""
schemas.py — canonical data contracts for the entire system.

Every message that crosses a module boundary is validated here.
Add fields freely, but never remove without bumping the version.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


# ─────────────────────────────────────────────
# 1.  Raw market data
# ─────────────────────────────────────────────

@dataclass
class BarData:
    """One OHLCV bar for one instrument."""
    symbol: str
    timestamp: int          # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float = 0.0     # bid-ask spread in price units


@dataclass
class UniverseSnapshot:
    """
    All bars for the current step across the traded universe.
    etf_bars  — ETF tickers used by the ETF Shock engine
    asset_bars — single-name / futures bars used by TCN + Graph engines
    """
    timestamp: int
    asset_bars: Dict[str, BarData]   # symbol -> BarData
    etf_bars:   Dict[str, BarData]   # symbol -> BarData


# ─────────────────────────────────────────────
# 2.  Strategy output (one per engine per step)
# ─────────────────────────────────────────────

@dataclass
class StrategySignal:
    """
    What each engine hands to the allocator.

    Fields
    ------
    engine_id      : "A" | "B" | "C"
    timestamp      : Unix ms
    eta_hat        : expected edge η̂_t^s  (annualised return estimate, e.g. 0.15 = 15%)
    utility_raw    : ν_t^s before budget solve (allocator may override)
    risk_score     : U_t^s  — variance / drawdown penalty term (non-negative)
    cost_estimate  : Ĉ_t^s — expected slippage + commission (non-negative, fraction of notional)
    positions      : symbol -> target weight within this engine's own book (-1 to +1)
    confidence     : scalar [0,1], how confident the model is (used in λ_U scaling)
    metadata       : free-form dict for debugging (e.g. graph edge weights, shock amplitudes)
    """
    engine_id: str
    timestamp: int
    eta_hat:      float
    utility_raw:  float
    risk_score:   float
    cost_estimate: float
    positions:    Dict[str, float] = field(default_factory=dict)
    confidence:   float = 1.0
    metadata:     Dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# 3.  Market state vector (allocator input)
# ─────────────────────────────────────────────

@dataclass
class MarketState:
    """
    z_t — raw market state from which s_alloc_t is inferred.

    All values are scalars unless noted.
    Engines and the allocator both read this; it is computed once per bar
    by sovereign_allocator.utils.state_builder.
    """
    timestamp: int
    # Volatility
    realized_vol_1d: float       # 1-day realised vol (annualised)
    realized_vol_5d: float       # 5-day realised vol (annualised)
    vix_proxy: float             # e.g. ATR-based fear gauge if no VIX feed

    # Cross-sectional dispersion
    cross_asset_dispersion: float   # std-dev of returns across the universe

    # Correlation regime
    avg_pairwise_corr: float        # rolling mean pairwise correlation

    # Breadth
    pct_above_sma20: float          # fraction of assets above their 20-bar SMA

    # Trend
    trend_strength: float           # e.g. ADX or slope of a linear fit, [0,1]

    # Liquidity / cost
    avg_spread_bps: float           # avg bid-ask spread in basis points
    avg_volume_ratio: float         # current volume / 20-bar avg volume

    # Optional macro overlay
    macro_surprise: float = 0.0     # standardised macro surprise index (0 if unavailable)


@dataclass
class AllocatorState:
    """
    s_alloc_t — the compact condition vector the allocator uses.
    Derived from MarketState by utils.state_builder.build_alloc_state().
    """
    timestamp: int
    volatility_regime: float    # [0,1]  0=calm, 1=stressed
    dispersion_regime: float    # [0,1]  0=compressed, 1=dispersed
    correlation_regime: float   # [0,1]  0=decorrelated, 1=highly correlated
    breadth_score:      float   # [0,1]  fraction above SMA20
    trend_score:        float   # [0,1]
    liquidity_score:    float   # [0,1]  1=liquid, 0=illiquid
    macro_score:        float   # [-1,1] negative=risk-off, positive=risk-on


# ─────────────────────────────────────────────
# 4.  Allocator output
# ─────────────────────────────────────────────

@dataclass
class BudgetAllocation:
    """
    b_t* — optimal strategy budgets from the QP solve.

    Fields
    ------
    budgets        : engine_id -> fraction of total capital [0,1], sums to ≤ 1
    cash_fraction  : 1 - sum(budgets), held in cash / margin buffer
    gamma_t        : risk-aversion parameter used this step
    lambda_turn    : turnover penalty used this step
    utility_scores : engine_id -> ν_t^s (for logging / dashboards)
    solve_status   : "optimal" | "infeasible" | "fallback"
    kill_switch    : True if governance layer forced a full de-risk
    """
    timestamp: int
    budgets:        Dict[str, float]
    cash_fraction:  float
    gamma_t:        float
    lambda_turn:    float
    utility_scores: Dict[str, float]
    solve_status:   str = "optimal"
    kill_switch:    bool = False
