"""
run_backtest.py — Quick synthetic backtest to validate the full pipeline.

Generates 500 bars of random-walk data across the universe and runs the
allocator. Prints a summary of budget allocations, utility scores, and
governance events.

Usage:
    python run_backtest.py
    python run_backtest.py --bars 1000 --seed 42
"""

from __future__ import annotations
import argparse
import numpy as np
from collections import defaultdict

from sovereign_allocator.data.schemas import BarData, UniverseSnapshot
from sovereign_allocator.engines.tcn_engine import TCNEngine
from sovereign_allocator.engines.graph_diffusion_engine import GraphDiffusionEngine
from sovereign_allocator.engines.etf_shock_engine import ETFShockEngine
from sovereign_allocator.allocator.portfolio import DynamicPortfolioAllocator
from sovereign_allocator.utils.state_builder import StateBuilder


ASSET_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
ETF_SYMBOLS   = ["SPY", "QQQ", "HYG", "TLT"]
ALL_SYMBOLS   = ASSET_SYMBOLS + ETF_SYMBOLS


def _random_walk_snapshot(t: int, prices: dict, rng: np.random.Generator) -> UniverseSnapshot:
    """One bar of a correlated random walk."""
    # Simulate mild correlation via a shared market factor
    market_factor = rng.normal(0, 0.005)

    def advance(sym: str, prev: float, idio_vol: float = 0.003) -> BarData:
        ret   = market_factor * 0.5 + rng.normal(0, idio_vol)
        close = prev * (1 + ret)
        hi    = close * (1 + abs(rng.normal(0, 0.001)))
        lo    = close * (1 - abs(rng.normal(0, 0.001)))
        return BarData(
            symbol=sym, timestamp=t,
            open=prev, high=hi, low=lo, close=close,
            volume=rng.uniform(5e6, 5e7),
            spread=rng.uniform(0.00005, 0.0005),
        )

    for sym in ALL_SYMBOLS:
        bar = advance(sym, prices[sym])
        prices[sym] = bar.close

    return UniverseSnapshot(
        timestamp=t,
        asset_bars={s: advance(s, prices[s]) for s in ASSET_SYMBOLS},
        etf_bars={s: advance(s, prices[s]) for s in ETF_SYMBOLS},
    )


def run(n_bars: int = 500, seed: int = 0):
    rng    = np.random.default_rng(seed)
    prices = {s: 100.0 + rng.uniform(-5, 5) for s in ALL_SYMBOLS}

    sb        = StateBuilder(symbols=ALL_SYMBOLS)
    allocator = DynamicPortfolioAllocator(
        engines=[
            TCNEngine(symbols=ASSET_SYMBOLS),
            GraphDiffusionEngine(symbols=ASSET_SYMBOLS),
            ETFShockEngine(asset_symbols=ASSET_SYMBOLS, etf_symbols=ETF_SYMBOLS),
        ],
        state_builder=sb,
    )

    budget_history = defaultdict(list)
    utility_history = defaultdict(list)
    kill_events = 0
    fallback_events = 0

    print(f"\n{'─'*60}")
    print(f"  Sovereign Allocator — Synthetic Backtest ({n_bars} bars)")
    print(f"{'─'*60}")
    print(f"  Engines : A=TCN  B=GraphDiffusion  C=ETFShock")
    print(f"  Symbols : {ASSET_SYMBOLS}")
    print(f"{'─'*60}\n")

    for i in range(n_bars):
        t    = 1_700_000_000_000 + i * 60_000
        snap = _random_walk_snapshot(t, prices, rng)
        alloc, positions = allocator.step(snap)

        for eid, b in alloc.budgets.items():
            budget_history[eid].append(b)
        for eid, u in alloc.utility_scores.items():
            utility_history[eid].append(u)

        if alloc.kill_switch:
            kill_events += 1
        if alloc.solve_status == "fallback":
            fallback_events += 1

        # Print a sample every 100 bars
        if (i + 1) % 100 == 0:
            b_str = "  ".join(f"{e}={v:.2%}" for e, v in alloc.budgets.items())
            print(f"  Bar {i+1:4d} | budgets: {b_str} | cash={alloc.cash_fraction:.2%} | {alloc.solve_status}")

    # Summary
    print(f"\n{'─'*60}")
    print("  BUDGET SUMMARY (mean over all bars)")
    for eid in ["A", "B", "C"]:
        arr = np.array(budget_history[eid])
        print(f"    Engine {eid}: mean={arr.mean():.2%}  std={arr.std():.2%}  "
              f"min={arr.min():.2%}  max={arr.max():.2%}")
    print(f"\n  Kill-switch events : {kill_events}")
    print(f"  Fallback solves    : {fallback_events}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(n_bars=args.bars, seed=args.seed)
