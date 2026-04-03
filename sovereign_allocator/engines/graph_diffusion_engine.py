"""
Engine B — Graph Diffusion Strategy (micro / cross-sectional)
=============================================================

What it does
------------
Treats the asset universe as a graph G = (V, E) where:
  - Nodes V = individual assets / instruments
  - Edges E = relationships (correlation, co-movement, sector linkage)

A shock arriving at one node (e.g. a large move in AAPL) propagates
to its neighbours via the graph Laplacian:

    H_t = (I + α * Ã)^K  @ x_t

where:
  Ã = symmetrically normalised adjacency matrix (correlation-based)
  α = diffusion coefficient  (how fast shocks travel)
  K = diffusion steps        (depth of neighbourhood)
  x_t = raw return / shock vector at time t

The diffused vector H_t is then used to predict near-term directional
bias for each asset: a positive H_t[i] means asset i is expected to
be dragged upward by its neighbours' positive moves.

Online graph update
-------------------
The adjacency matrix is recomputed every `graph_update_freq` bars using
the rolling correlation matrix.  A sparse threshold `min_edge_weight`
keeps the graph interpretable and avoids over-diffusion in dense markets.

This engine captures the MICRO / CROSS-SECTIONAL edge:
  - Idiosyncratic shocks that ripple through related names
  - Sector / factor contagion not explained by a single market factor
  - Momentum spillovers between correlated instruments
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np

from sovereign_allocator.data.schemas import (
    UniverseSnapshot, MarketState, AllocatorState, StrategySignal
)
from sovereign_allocator.engines.base import BaseEngine


class GraphDiffusionEngine(BaseEngine):
    """
    Parameters
    ----------
    symbols            : universe of assets (nodes)
    corr_window        : bars used to compute rolling correlation for edges
    graph_update_freq  : re-build the graph every N bars
    diffusion_alpha    : α — shock propagation strength [0, 1]
    diffusion_steps    : K — neighbourhood depth
    min_edge_weight    : threshold below which edges are set to zero (sparsity)
    position_cap       : max abs weight per symbol
    """

    ENGINE_ID = "B"

    def __init__(
        self,
        symbols: List[str],
        corr_window: int = 60,
        graph_update_freq: int = 20,
        diffusion_alpha: float = 0.3,
        diffusion_steps: int = 2,
        min_edge_weight: float = 0.3,
        position_cap: float = 0.20,
        lambda_u: float = 1.2,
        lambda_c: float = 0.5,
    ):
        super().__init__(engine_id=self.ENGINE_ID, lambda_u=lambda_u, lambda_c=lambda_c)
        self.symbols           = symbols
        self.n                 = len(symbols)
        self.sym_idx           = {s: i for i, s in enumerate(symbols)}
        self.corr_window       = corr_window
        self.graph_update_freq = graph_update_freq
        self.diffusion_alpha   = diffusion_alpha
        self.diffusion_steps   = diffusion_steps
        self.min_edge_weight   = min_edge_weight
        self.position_cap      = position_cap

        # Rolling return matrix  (n_assets × window)
        self._return_buf: deque = deque(maxlen=corr_window)
        self._prev_closes: Dict[str, float] = {}

        # Graph state
        self._adj: Optional[np.ndarray] = None        # normalised adjacency
        self._diffusion_op: Optional[np.ndarray] = None  # (I + α*Ã)^K
        self._bar_count: int = 0

    # ──────────────────────────────────────────
    # Graph construction
    # ──────────────────────────────────────────

    def _update_graph(self, return_matrix: np.ndarray):
        """
        Rebuild the adjacency matrix from rolling correlations.
        return_matrix : shape (T, n_assets)
        """
        if return_matrix.shape[0] < 10:
            return

        corr = np.corrcoef(return_matrix.T)            # (n, n)
        # Threshold: only keep strong edges
        adj  = np.where(np.abs(corr) >= self.min_edge_weight, corr, 0.0)
        np.fill_diagonal(adj, 0.0)                      # no self-loops

        # Symmetric normalisation: Ã = D^{-1/2} A D^{-1/2}
        deg  = np.abs(adj).sum(axis=1)
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg + 1e-8), 0.0)
        D    = np.diag(d_inv_sqrt)
        adj_norm = D @ adj @ D

        # Diffusion operator: (I + α * Ã)^K
        I    = np.eye(self.n)
        op   = I + self.diffusion_alpha * adj_norm
        diffusion_op = np.linalg.matrix_power(op, self.diffusion_steps)

        self._adj          = adj_norm
        self._diffusion_op = diffusion_op

    # ──────────────────────────────────────────
    # Core signal logic
    # ──────────────────────────────────────────

    def compute_signal(
        self,
        snapshot: UniverseSnapshot,
        state: MarketState,
        alloc_state: Optional[AllocatorState] = None,
    ) -> StrategySignal:

        # 1. Collect returns this bar
        bar_returns = np.zeros(self.n)
        for symbol, i in self.sym_idx.items():
            bar = snapshot.asset_bars.get(symbol)
            if bar is None:
                continue
            prev = self._prev_closes.get(symbol, bar.open)
            log_ret = np.log(bar.close / (prev + 1e-9))
            bar_returns[i]          = log_ret
            self._prev_closes[symbol] = bar.close

        self._return_buf.append(bar_returns.copy())
        self._bar_count += 1

        # 2. Periodically rebuild graph
        if self._bar_count % self.graph_update_freq == 0 and len(self._return_buf) >= 10:
            return_matrix = np.vstack(self._return_buf)
            self._update_graph(return_matrix)

        # 3. Diffuse shocks if we have a graph
        positions: Dict[str, float] = {}
        graph_signal = np.zeros(self.n)

        if self._diffusion_op is not None:
            # Shock vector = current bar's returns (normalised)
            shock_vol = np.std(bar_returns) + 1e-8
            x_t = bar_returns / shock_vol

            # Diffuse: H_t = Ω @ x_t
            H_t = self._diffusion_op @ x_t      # (n,)

            graph_signal = H_t
            # Scale to positions
            signal_abs_max = np.abs(H_t).max() + 1e-8
            for symbol, i in self.sym_idx.items():
                raw = H_t[i] / signal_abs_max * self.position_cap
                positions[symbol] = float(np.clip(raw, -self.position_cap, self.position_cap))

        # 4. Risk and cost
        # Risk rises when cross-asset correlation is high (contagion risk)
        risk_score = state.avg_pairwise_corr * state.realized_vol_5d
        cost_est   = state.avg_spread_bps / 10_000

        # Confidence: how dispersed the diffused signals are
        sig_std    = float(np.std(graph_signal)) if len(graph_signal) > 0 else 0.0
        confidence = min(sig_std * 5, 1.0)    # scale to [0,1]

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
                "n_edges": int(np.count_nonzero(self._adj)) // 2 if self._adj is not None else 0,
                "avg_diffused_signal": float(np.mean(np.abs(graph_signal))),
                "graph_update_bar": self._bar_count,
            },
        )
