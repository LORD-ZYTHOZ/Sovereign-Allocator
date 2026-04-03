"""
Sovereign Allocator — Multi-Strategy Dynamic Portfolio System
=============================================================
Three orthogonal trading engines unified under a regime-aware,
utility-maximising quadratic-program allocator.

Engines
-------
A  TCN (Causal Temporal Convolutional Network)  — temporal / time-series edge
B  Graph Diffusion                              — cross-asset spillover edge
C  ETF Shock Propagation                        — macro flow-driven edge

Allocator
---------
Reads a market state vector s_alloc_t, scores each engine with a
proprietary utility ν_t^s, then solves a QP over strategy budgets b_t.

Governance
----------
Hard kill-switches and drawdown rules that override the QP.
"""

from sovereign_allocator.data.schemas import BarData, StrategySignal, AllocatorState
from sovereign_allocator.allocator.portfolio import DynamicPortfolioAllocator

__all__ = ["DynamicPortfolioAllocator", "BarData", "StrategySignal", "AllocatorState"]
__version__ = "0.1.0"
