"""
portfolio.py — DynamicPortfolioAllocator

This is the top-level orchestrator. Each bar it:

1.  Receives the UniverseSnapshot and calls each engine's .step()
2.  Builds the MarketState → AllocatorState via state_builder
3.  Updates the online covariance estimator with this bar's engine PnLs
4.  Scores each engine using its utility_raw from the signal
5.  Runs the QP solver to get optimal budgets b_t*
6.  Passes budgets through the GovernanceLayer (kill-switches, drawdown rules)
7.  Computes final combined positions across all engines
8.  Returns a BudgetAllocation + the combined position dict

Usage
-----
    from sovereign_allocator import DynamicPortfolioAllocator
    from sovereign_allocator.engines import TCNEngine, GraphDiffusionEngine, ETFShockEngine
    from sovereign_allocator.utils.state_builder import StateBuilder

    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    etfs    = ["SPY", "QQQ", "HYG", "TLT"]

    allocator = DynamicPortfolioAllocator(
        engines=[
            TCNEngine(symbols=symbols),
            GraphDiffusionEngine(symbols=symbols),
            ETFShockEngine(asset_symbols=symbols, etf_symbols=etfs),
        ],
        state_builder=StateBuilder(symbols=symbols + etfs),
    )

    for snapshot in data_feed:
        allocation, positions = allocator.step(snapshot)
        # allocation.budgets  → {"A": 0.45, "B": 0.30, "C": 0.20}
        # positions           → {"EURUSD": 0.12, "XAUUSD": -0.05, ...}
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from sovereign_allocator.data.schemas import (
    UniverseSnapshot, MarketState, AllocatorState, BudgetAllocation, StrategySignal
)
from sovereign_allocator.engines.base import BaseEngine
from sovereign_allocator.allocator.covariance import OnlineCovarianceEstimator
from sovereign_allocator.allocator.qp_solver import QPAllocator
from sovereign_allocator.governance.kill_switch import GovernanceLayer
from sovereign_allocator.utils.state_builder import StateBuilder


class DynamicPortfolioAllocator:
    """
    Parameters
    ----------
    engines          : list of BaseEngine subclasses (A, B, C)
    state_builder    : StateBuilder instance
    gamma_base       : QP risk-aversion base
    lambda_turn      : QP turnover penalty base
    max_drawdown     : portfolio-level hard drawdown limit (fraction, e.g. 0.10 = 10%)
    per_engine_dd    : per-engine hard drawdown limit
    budget_cap       : max fraction any single engine can receive
    """

    def __init__(
        self,
        engines: List[BaseEngine],
        state_builder: StateBuilder,
        gamma_base: float = 2.0,
        lambda_turn: float = 0.10,
        max_drawdown: float = 0.10,
        per_engine_dd: float = 0.05,
        budget_cap: float = 0.80,
    ):
        self.engines       = {e.engine_id: e for e in engines}
        self.engine_ids    = [e.engine_id for e in engines]
        self.state_builder = state_builder

        # Sub-components
        self._cov_estimator = OnlineCovarianceEstimator(engine_ids=self.engine_ids)
        self._qp            = QPAllocator(
            engine_ids=self.engine_ids,
            gamma_base=gamma_base,
            lambda_turn=lambda_turn,
            budget_cap=budget_cap,
        )
        self._governance    = GovernanceLayer(
            engine_ids=self.engine_ids,
            max_portfolio_drawdown=max_drawdown,
            max_per_engine_drawdown=per_engine_dd,
        )

        # State tracking
        self._last_signals: Dict[str, StrategySignal] = {}
        self._peak_nav: float = 1.0
        self._nav: float = 1.0

    # ──────────────────────────────────────────
    # Main step
    # ──────────────────────────────────────────

    def step(
        self,
        snapshot: UniverseSnapshot,
    ) -> Tuple[BudgetAllocation, Dict[str, float]]:
        """
        Process one bar and return (BudgetAllocation, combined_positions).
        """

        # 1. Build market state
        market_state = self.state_builder.build_market_state(snapshot)
        alloc_state  = self.state_builder.build_alloc_state(market_state)

        # 2. Run each engine
        signals: Dict[str, StrategySignal] = {}
        for eid, engine in self.engines.items():
            signals[eid] = engine.step(snapshot, market_state, alloc_state)
        self._last_signals = signals

        # 3. Update covariance with realized PnLs (from engine base class history)
        pnl_vec = {eid: sig.eta_hat / 252 for eid, sig in signals.items()}
        self._cov_estimator.update(pnl_vec)
        sigma = self._cov_estimator.get()

        # 4. Utility scores
        utility_scores = {eid: sig.utility_raw for eid, sig in signals.items()}

        # 5. Governance pre-check (may zero out individual engines)
        utility_scores, engine_enabled = self._governance.pre_check(
            utility_scores=utility_scores,
            signals=signals,
            alloc_state=alloc_state,
        )

        # 6. QP solve
        budgets, cash_fraction, status = self._qp.solve(
            utility_scores=utility_scores,
            sigma=sigma,
            volatility_regime=alloc_state.volatility_regime,
            liquidity_score=alloc_state.liquidity_score,
        )

        # 7. Governance post-check (portfolio-level kill-switch)
        kill_switch = self._governance.post_check(self._nav, self._peak_nav)
        if kill_switch:
            budgets       = {eid: 0.0 for eid in self.engine_ids}
            cash_fraction = 1.0
            status        = "kill_switch"

        # 8. Build combined positions
        combined_positions = self._combine_positions(budgets, signals)

        # 9. Build allocation record
        allocation = BudgetAllocation(
            timestamp=snapshot.timestamp,
            budgets=budgets,
            cash_fraction=cash_fraction,
            gamma_t=self._qp.gamma_base,
            lambda_turn=self._qp.lambda_turn,
            utility_scores=utility_scores,
            solve_status=status,
            kill_switch=kill_switch,
        )

        return allocation, combined_positions

    # ──────────────────────────────────────────
    # Position aggregation
    # ──────────────────────────────────────────

    def _combine_positions(
        self,
        budgets: Dict[str, float],
        signals: Dict[str, StrategySignal],
    ) -> Dict[str, float]:
        """
        Aggregate per-engine positions, scaled by their budget fractions.

        For each symbol, the combined position is the budget-weighted sum
        of each engine's position in that symbol.
        """
        combined: Dict[str, float] = {}

        for eid, signal in signals.items():
            budget = budgets.get(eid, 0.0)
            for symbol, pos in signal.positions.items():
                combined[symbol] = combined.get(symbol, 0.0) + budget * pos

        # Clip combined positions to [-1, +1] for sanity
        return {sym: float(np.clip(p, -1.0, 1.0)) for sym, p in combined.items()}

    def update_nav(self, new_nav: float):
        """Call this after each bar with the current portfolio NAV."""
        self._nav = new_nav
        if new_nav > self._peak_nav:
            self._peak_nav = new_nav
        self._governance.update_nav(new_nav, self._peak_nav)
