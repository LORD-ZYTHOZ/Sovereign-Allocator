"""
Smoke tests — verify the full pipeline runs end-to-end without crashing.
No real data needed; we generate synthetic bars.

Run with:
    python -m pytest tests/ -v
"""

import time
import numpy as np
import pytest

from sovereign_allocator.data.schemas import BarData, UniverseSnapshot
from sovereign_allocator.engines.tcn_engine import TCNEngine
from sovereign_allocator.engines.graph_diffusion_engine import GraphDiffusionEngine
from sovereign_allocator.engines.etf_shock_engine import ETFShockEngine
from sovereign_allocator.allocator.portfolio import DynamicPortfolioAllocator
from sovereign_allocator.utils.state_builder import StateBuilder


ASSET_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
ETF_SYMBOLS   = ["SPY", "QQQ", "HYG", "TLT"]
ALL_SYMBOLS   = ASSET_SYMBOLS + ETF_SYMBOLS


def _make_snapshot(t: int, rng: np.random.Generator) -> UniverseSnapshot:
    """Generate a synthetic snapshot with random walks."""
    def make_bar(sym: str) -> BarData:
        close = 100.0 * (1 + rng.normal(0, 0.001))
        return BarData(
            symbol=sym,
            timestamp=t,
            open=close * (1 - abs(rng.normal(0, 0.0005))),
            high=close * (1 + abs(rng.normal(0, 0.001))),
            low=close  * (1 - abs(rng.normal(0, 0.001))),
            close=close,
            volume=rng.uniform(1e6, 1e7),
            spread=rng.uniform(0.0001, 0.001),
        )

    return UniverseSnapshot(
        timestamp=t,
        asset_bars={s: make_bar(s) for s in ASSET_SYMBOLS},
        etf_bars={s: make_bar(s) for s in ETF_SYMBOLS},
    )


@pytest.fixture
def allocator():
    sb = StateBuilder(symbols=ALL_SYMBOLS)
    return DynamicPortfolioAllocator(
        engines=[
            TCNEngine(symbols=ASSET_SYMBOLS),
            GraphDiffusionEngine(symbols=ASSET_SYMBOLS),
            ETFShockEngine(asset_symbols=ASSET_SYMBOLS, etf_symbols=ETF_SYMBOLS),
        ],
        state_builder=sb,
    )


class TestSmokeRun:

    def test_single_step(self, allocator):
        rng = np.random.default_rng(42)
        snap = _make_snapshot(t=1_000_000, rng=rng)
        alloc, positions = allocator.step(snap)
        assert alloc is not None
        assert isinstance(positions, dict)

    def test_budget_sums_to_leq_one(self, allocator):
        rng = np.random.default_rng(42)
        for i in range(5):
            snap = _make_snapshot(t=i * 60_000, rng=rng)
            alloc, _ = allocator.step(snap)
        total = sum(alloc.budgets.values())
        assert total <= 1.01, f"Budgets sum to {total:.4f} > 1"

    def test_positions_bounded(self, allocator):
        rng = np.random.default_rng(7)
        for i in range(30):
            snap = _make_snapshot(t=i * 60_000, rng=rng)
            alloc, positions = allocator.step(snap)
        for sym, pos in positions.items():
            assert -1.0 <= pos <= 1.0, f"{sym} position {pos} out of [-1,1]"

    def test_kill_switch_fires(self, allocator):
        """Simulate a 15% drawdown — kill switch should activate."""
        allocator.update_nav(0.85)   # 15% drawdown from peak=1.0
        rng = np.random.default_rng(1)
        snap = _make_snapshot(t=999_999, rng=rng)
        alloc, _ = allocator.step(snap)
        assert alloc.kill_switch is True
        assert all(v == 0.0 for v in alloc.budgets.values())

    def test_100_bar_run(self, allocator):
        """Stress test: 100 bars, no exceptions."""
        rng = np.random.default_rng(99)
        for i in range(100):
            snap = _make_snapshot(t=i * 60_000, rng=rng)
            alloc, positions = allocator.step(snap)
        assert alloc.solve_status in ("optimal", "fallback", "kill_switch")

    def test_engine_outputs(self):
        """Each engine should produce a StrategySignal independently."""
        rng   = np.random.default_rng(5)
        sb    = StateBuilder(symbols=ALL_SYMBOLS)
        tcn   = TCNEngine(symbols=ASSET_SYMBOLS)
        graph = GraphDiffusionEngine(symbols=ASSET_SYMBOLS)
        etf   = ETFShockEngine(asset_symbols=ASSET_SYMBOLS, etf_symbols=ETF_SYMBOLS)

        for i in range(20):
            snap  = _make_snapshot(t=i * 60_000, rng=rng)
            ms    = sb.build_market_state(snap)
            als   = sb.build_alloc_state(ms)
            sig_a = tcn.step(snap, ms, als)
            sig_b = graph.step(snap, ms, als)
            sig_c = etf.step(snap, ms, als)

        assert sig_a.engine_id == "A"
        assert sig_b.engine_id == "B"
        assert sig_c.engine_id == "C"
        assert isinstance(sig_a.utility_raw, float)
