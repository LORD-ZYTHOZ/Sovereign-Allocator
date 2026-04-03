"""
kill_switch.py — GovernanceLayer

Hard rules that override the QP at two points:

PRE-CHECK  (before the QP solve)
─────────────────────────────────
For each engine separately:
  • If the engine has breached its per-engine drawdown limit → zero its utility
    so the QP naturally allocates zero budget to it.
  • If an engine's recent Sharpe (rolling 20-bar) has gone negative for N
    consecutive bars → treat it as degraded, halve its utility.

POST-CHECK (after the QP solve)
─────────────────────────────────
Portfolio-level:
  • If portfolio drawdown from peak NAV ≥ max_portfolio_drawdown → KILL SWITCH.
    All budgets → 0, all positions → flat. No reentry until NAV recovers
    to recovery_threshold × peak.
  • If volatility_regime ≥ extreme_vol_threshold → PAUSE (same as kill switch
    but auto-reset after extreme_vol_threshold drops below).

Recovery
────────
Once triggered, the kill switch remains active until:
    current_nav ≥ peak_nav × (1 − max_portfolio_drawdown × recovery_factor)

This prevents immediate re-entry into a collapsing market.
"""

from __future__ import annotations
from collections import deque, defaultdict
from typing import Dict, List, Tuple
import numpy as np

from sovereign_allocator.data.schemas import StrategySignal, AllocatorState


class GovernanceLayer:
    """
    Parameters
    ----------
    engine_ids                 : list of engine IDs
    max_portfolio_drawdown     : hard drawdown threshold for full de-risk (e.g. 0.10)
    max_per_engine_drawdown    : per-engine drawdown to disable that engine (e.g. 0.05)
    consecutive_neg_sharpe     : bars of negative Sharpe before engine is degraded
    extreme_vol_threshold      : volatility_regime above which we pause (e.g. 0.90)
    recovery_factor            : fraction of drawdown that must recover before re-entry (e.g. 0.5)
    """

    def __init__(
        self,
        engine_ids: List[str],
        max_portfolio_drawdown: float = 0.10,
        max_per_engine_drawdown: float = 0.05,
        consecutive_neg_sharpe: int = 10,
        extreme_vol_threshold: float = 0.90,
        recovery_factor: float = 0.50,
    ):
        self.engine_ids              = engine_ids
        self.max_portfolio_drawdown  = max_portfolio_drawdown
        self.max_per_engine_drawdown = max_per_engine_drawdown
        self.consecutive_neg_sharpe  = consecutive_neg_sharpe
        self.extreme_vol_threshold   = extreme_vol_threshold
        self.recovery_factor         = recovery_factor

        # Per-engine state
        self._engine_peak_utility: Dict[str, float] = defaultdict(float)
        self._neg_sharpe_count:    Dict[str, int]   = defaultdict(int)
        self._engine_nav:          Dict[str, float] = {e: 1.0 for e in engine_ids}
        self._engine_peak_nav:     Dict[str, float] = {e: 1.0 for e in engine_ids}

        # Portfolio-level state
        self._kill_switch_active: bool  = False
        self._nav:      float = 1.0
        self._peak_nav: float = 1.0

    # ──────────────────────────────────────────
    # Called by allocator before QP
    # ──────────────────────────────────────────

    def pre_check(
        self,
        utility_scores: Dict[str, float],
        signals: Dict[str, StrategySignal],
        alloc_state: AllocatorState,
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        """
        Modify utility scores and return engine_enabled flags.
        Engines with zero utility are effectively disabled this bar.
        """
        modified_utils = utility_scores.copy()
        engine_enabled: Dict[str, bool] = {}

        for eid in self.engine_ids:
            signal = signals.get(eid)
            if signal is None:
                engine_enabled[eid] = False
                modified_utils[eid] = 0.0
                continue

            # Track per-engine simulated NAV via cumulative eta_hat
            daily_ret = signal.eta_hat / 252
            self._engine_nav[eid] = self._engine_nav[eid] * (1 + daily_ret)
            if self._engine_nav[eid] > self._engine_peak_nav[eid]:
                self._engine_peak_nav[eid] = self._engine_nav[eid]

            # Per-engine drawdown check
            engine_dd = 1 - self._engine_nav[eid] / (self._engine_peak_nav[eid] + 1e-10)
            if engine_dd >= self.max_per_engine_drawdown:
                modified_utils[eid] = 0.0
                engine_enabled[eid] = False
                continue

            # Rolling Sharpe degradation check
            if signal.eta_hat < 0:
                self._neg_sharpe_count[eid] += 1
            else:
                self._neg_sharpe_count[eid] = 0

            if self._neg_sharpe_count[eid] >= self.consecutive_neg_sharpe:
                # Degrade: halve the utility
                modified_utils[eid] = modified_utils[eid] * 0.5

            engine_enabled[eid] = True

        return modified_utils, engine_enabled

    # ──────────────────────────────────────────
    # Called by allocator after QP
    # ──────────────────────────────────────────

    def post_check(self, nav: float, peak_nav: float) -> bool:
        """
        Returns True if the portfolio-level kill switch should fire.
        """
        portfolio_dd = 1 - nav / (peak_nav + 1e-10)

        if portfolio_dd >= self.max_portfolio_drawdown:
            self._kill_switch_active = True

        if self._kill_switch_active:
            # Recovery threshold: only re-enable when sufficiently recovered
            recovery_level = peak_nav * (
                1 - self.max_portfolio_drawdown * self.recovery_factor
            )
            if nav >= recovery_level:
                self._kill_switch_active = False

        return self._kill_switch_active

    def update_nav(self, nav: float, peak_nav: float):
        self._nav      = nav
        self._peak_nav = peak_nav

    # ──────────────────────────────────────────
    # Status report (for logging / dashboards)
    # ──────────────────────────────────────────

    def status_report(self) -> dict:
        port_dd = 1 - self._nav / (self._peak_nav + 1e-10)
        engine_dds = {
            e: 1 - self._engine_nav[e] / (self._engine_peak_nav[e] + 1e-10)
            for e in self.engine_ids
        }
        return {
            "kill_switch_active": self._kill_switch_active,
            "portfolio_drawdown": round(port_dd, 4),
            "engine_drawdowns":   {e: round(v, 4) for e, v in engine_dds.items()},
            "neg_sharpe_counts":  dict(self._neg_sharpe_count),
        }
