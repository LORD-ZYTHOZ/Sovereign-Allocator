# Sovereign Allocator

A multi-strategy, regime-aware dynamic portfolio system. Three orthogonal trading engines are unified under a utility-maximising quadratic-program allocator with a hard governance layer.

Built with $100,000 of live capital behind it.

---

## What it does

Most algorithmic trading systems rely on one model. If that model breaks (regime shift, data issue, edge decay), you lose. The Sovereign Allocator runs **three independent engines** that find edge in completely different ways, then uses a **mathematical optimiser** to decide in real-time how much capital each engine deserves — punishing risk, cost, and poor recent performance.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Market Data Feed                       │
│              UniverseSnapshot (every bar)                 │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   State Builder                          │
│  Raw bars → MarketState z_t → AllocatorState s_alloc_t  │
│  volatility · dispersion · correlation · breadth ·      │
│  trend · liquidity · macro                              │
└──────┬───────────────┬────────────────┬─────────────────┘
       │               │                │
       ▼               ▼                ▼
┌──────────┐   ┌──────────────┐   ┌────────────────┐
│ Engine A │   │   Engine B   │   │   Engine C     │
│   TCN    │   │    Graph     │   │   ETF Shock    │
│(temporal)│   │  Diffusion   │   │ (macro flows)  │
│          │   │(cross-asset) │   │                │
│ η̂, U, Ĉ │   │  η̂, U, Ĉ    │   │   η̂, U, Ĉ    │
└────┬─────┘   └──────┬───────┘   └──────┬─────────┘
     │                │                   │
     └────────────────┴───────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│              Dynamic Portfolio Allocator                  │
│                                                          │
│  1. Utility score per engine:                            │
│       ν_t^s = η̂_t^s − λ_U · U_t^s − λ_C · Ĉ_t^s      │
│                                                          │
│  2. Online covariance Σ_t^{slv} (EWMA + shrinkage)      │
│                                                          │
│  3. QP solve:                                            │
│     b* = argmax [ ν^T b − (γ/2)b^T Σ b − λ‖b−b_prev‖₁ ]│
│     s.t. 1^T b ≤ 1,  b ≥ 0                              │
│                                                          │
│  4. Governance pre/post check                            │
└──────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│  Combined positions = Σ_s budget_s × engine_s.positions  │
└──────────────────────────────────────────────────────────┘
```

---

## The Three Engines

### Engine A — Causal TCN (Temporal / Time-Series Edge)

**What it captures:** Patterns in a single asset's own price history.

A [causal dilated temporal convolutional network](https://arxiv.org/abs/1803.01271) reads the last `L` bars of returns and produces a probability distribution over {short, flat, long} for the next bar:

```
P(r_{t+1} | r_t, r_{t-1}, ..., r_{t-L})
```

Causal means no look-ahead leakage — each output depends only on past inputs. Dilated convolutions exponentially expand the receptive field without adding parameters.

**Why it's useful:** Captures momentum, mean-reversion regimes, and autocorrelation structures that purely statistical models miss.

**Output:** Directional position for each symbol, scaled by model confidence.

**Stub mode:** Out of the box, a momentum-based proxy (`_CausalTCNStub`) is used. Swap in your trained PyTorch / MLX model via the `model=` parameter.

---

### Engine B — Graph Diffusion (Cross-Asset / Micro Edge)

**What it captures:** How a shock in one asset ripples through its correlated neighbours.

The asset universe is modelled as a graph G = (V, E):
- **Nodes** = individual assets
- **Edges** = rolling correlations (sparse, only edges above `min_edge_weight`)

Shocks propagate via the diffusion operator:

```
H_t = (I + α · Ã)^K  ·  x_t
```

Where:
- `Ã` = symmetrically normalised adjacency matrix
- `α` = diffusion coefficient (how fast shocks travel between nodes)
- `K` = depth of neighbourhood to look (diffusion steps)
- `x_t` = current bar's return vector (shock vector)

A positive `H_t[i]` means asset `i` is being dragged upward by its neighbours.

**Why it's useful:** Captures contagion, sector spillovers, and co-movement not explained by a single market factor. A large move in a liquid name (e.g. AAPL) diffuses to related illiquid names before they respond.

**Graph update:** Rebuilt every `graph_update_freq` bars using rolling correlation.

---

### Engine C — ETF Shock Propagation (Macro / Top-Down Edge)

**What it captures:** ETF flows and macro shocks propagating into individual assets.

**Pipeline:**

1. **Shock detection** — for each ETF, compute a rolling z-score of its return:
   ```
   z_t^{ETF} = (r_t^{ETF} − μ) / σ
   ```
   A shock is declared when `|z| > threshold` (default: 2.0 standard deviations).

2. **Beta estimation** — rolling OLS gives each asset's loading on each ETF:
   ```
   β_{asset, ETF} = Cov(r^asset, r^ETF) / Var(r^ETF)
   ```

3. **Propagation** — when a shock fires, inject a directional drift into each asset:
   ```
   drift_{t, asset} = Σ_ETF  β_{asset, ETF} × shock_amplitude
   ```

4. **Decay** — the injected signal tapers linearly over `signal_decay` bars. A shock from 3 bars ago has less influence than a fresh one.

**Why it's useful:** Captures top-down flow effects — a large inflow into SPY lifts its constituents; a HYG dump signals credit stress that cascades into high-beta equities.

---

## The Allocator

### Utility Score

For each engine `s` at time `t`:

```
ν_t^s = η̂_t^s − λ_U · U_t^s − λ_C · Ĉ_t^s
```

| Term | Meaning |
|------|---------|
| `η̂_t^s` | Expected edge — EWMA of realised PnL, annualised. Stronger recent performance → higher η̂. |
| `U_t^s` | Risk penalty — a function of realised volatility and regime. High volatility → penalise more. |
| `Ĉ_t^s` | Cost / slippage estimate — proportional to bid-ask spread and position churn. |
| `λ_U`, `λ_C` | Penalty weights — tunable per engine. |

An engine with a positive edge but high risk and cost can still have a negative utility, causing the QP to allocate it less budget.

### Quadratic Programme

The allocator solves for optimal strategy budgets `b_t*`:

```
b* = argmax_b  [ ν^T b  −  (γ/2) b^T Σ b  −  λ_turn ‖b − b_{t-1}‖₁ ]

subject to:  1^T b ≤ 1    (budgets sum to at most 100%)
             b ≥ 0         (no negative allocation to an engine)
```

| Term | Effect |
|------|--------|
| `ν^T b` | Reward: allocate more to higher-utility engines |
| `(γ/2) b^T Σ b` | Risk penalty: penalise correlated engine exposures (Σ = strategy covariance) |
| `λ_turn ‖b − b_{t-1}‖₁` | Turnover penalty: avoid thrashing allocations every bar |

**Adaptive parameters:**
- `γ_t` scales with `volatility_regime` — more risk-averse in stressed markets
- `λ_turn` scales with `(1 − liquidity_score)` — higher turnover cost when illiquid

The L1 penalty is linearised via auxiliary slack variables and solved with SLSQP (scipy). No external QP solver needed.

### Online Covariance (Σ)

Σ is the 3×3 strategy-level covariance matrix estimated from each engine's realised PnL stream using exponentially weighted moving average (EWMA). Ledoit-Wolf constant-correlation shrinkage keeps it well-conditioned even when the window is short.

---

## Governance Layer

Hard rules that override the QP — the maths never gets final say.

### Pre-check (per engine, before QP)

| Rule | Action |
|------|--------|
| Engine drawdown ≥ `per_engine_dd` (5%) | Set engine utility = 0, QP allocates nothing |
| Engine negative Sharpe for N consecutive bars | Halve the engine's utility |

### Post-check (portfolio level, after QP)

| Rule | Action |
|------|--------|
| Portfolio drawdown from peak ≥ `max_drawdown` (10%) | **Kill switch**: all budgets → 0, all positions → flat |
| Recovery threshold not yet reached | Kill switch remains active |

Recovery is defined as: `NAV ≥ peak × (1 − max_drawdown × recovery_factor)`. This prevents re-entering a collapsing market immediately after a brief bounce.

---

## File Structure

```
the-code/
├── sovereign_allocator/
│   ├── data/
│   │   └── schemas.py           # BarData, UniverseSnapshot, StrategySignal,
│   │                            #   MarketState, AllocatorState, BudgetAllocation
│   ├── engines/
│   │   ├── base.py              # BaseEngine ABC — step(), EWMA η̂, realized PnL
│   │   ├── tcn_engine.py        # Engine A: Causal TCN
│   │   ├── graph_diffusion_engine.py  # Engine B: Graph Diffusion
│   │   └── etf_shock_engine.py  # Engine C: ETF Shock Propagation
│   ├── allocator/
│   │   ├── covariance.py        # Online EWMA covariance + Ledoit-Wolf shrinkage
│   │   ├── qp_solver.py         # Quadratic programme (SLSQP via scipy)
│   │   └── portfolio.py         # DynamicPortfolioAllocator — wires everything
│   ├── governance/
│   │   └── kill_switch.py       # GovernanceLayer — per-engine + portfolio rules
│   └── utils/
│       └── state_builder.py     # Raw bars → MarketState → AllocatorState
├── configs/
│   └── default.yaml             # All tunable hyperparameters
├── tests/
│   └── test_smoke.py            # 6 smoke tests (all passing)
├── run_backtest.py               # Synthetic 500-bar backtest demo
├── requirements.txt
└── setup.py
```

---

## Quick Start

```bash
# Install
pip install -e .

# Run the smoke tests
python -m pytest tests/ -v

# Run a synthetic 500-bar backtest
python run_backtest.py

# Custom run
python run_backtest.py --bars 1000 --seed 42
```

---

## Wiring to Your Data Feed

The entry point is `DynamicPortfolioAllocator.step(snapshot)`:

```python
from sovereign_allocator import DynamicPortfolioAllocator
from sovereign_allocator.engines import TCNEngine, GraphDiffusionEngine, ETFShockEngine
from sovereign_allocator.utils.state_builder import StateBuilder
from sovereign_allocator.data.schemas import BarData, UniverseSnapshot

ASSETS = ["EURUSD", "XAUUSD", "NAS100", "US30"]
ETFS   = ["SPY", "QQQ", "HYG", "TLT"]

allocator = DynamicPortfolioAllocator(
    engines=[
        TCNEngine(symbols=ASSETS),
        GraphDiffusionEngine(symbols=ASSETS),
        ETFShockEngine(asset_symbols=ASSETS, etf_symbols=ETFS),
    ],
    state_builder=StateBuilder(symbols=ASSETS + ETFS),
)

# In your bar loop:
for bar_data in your_feed:
    snapshot = UniverseSnapshot(
        timestamp=bar_data.timestamp,
        asset_bars={s: BarData(...) for s in ASSETS},
        etf_bars={e: BarData(...) for e in ETFS},
    )
    allocation, positions = allocator.step(snapshot)

    # allocation.budgets  → {"A": 0.42, "B": 0.31, "C": 0.22}
    # positions           → {"EURUSD": 0.14, "XAUUSD": -0.08, ...}
    # allocation.kill_switch → True if governance fired
    allocator.update_nav(your_current_nav)
```

---

## Plugging in a Trained TCN

Replace the stub with your PyTorch or MLX model:

```python
class MyTCNModel:
    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        # returns: 1-D array of log-returns (newest last)
        # must return: np.array([p_short, p_flat, p_long])
        ...

engine_a = TCNEngine(symbols=ASSETS, model=MyTCNModel())
```

---

## Configuration

All hyperparameters live in `configs/default.yaml`. Notable ones:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `gamma_base` | 2.0 | Base risk aversion in QP. Higher = more conservative. |
| `lambda_turn` | 0.10 | Turnover penalty. Higher = stickier allocations. |
| `max_drawdown` | 0.10 | Portfolio drawdown threshold for kill switch (10%). |
| `per_engine_dd` | 0.05 | Per-engine drawdown before engine is disabled (5%). |
| `diffusion_alpha` | 0.3 | Graph shock propagation speed. |
| `shock_threshold` | 2.0 | ETF z-score to declare a macro shock. |

---

## What's Left to Tune

These are the placeholder estimation pieces you'll calibrate on live data:

1. **`η̂` half-life** — the EWMA decay in `BaseEngine._estimate_eta()`. Default 20 bars. Shorter = more reactive; longer = more stable.
2. **`λ_U`, `λ_C` per engine** — how much you penalise risk vs cost for each strategy.
3. **Normalisation thresholds** in `StateBuilder.build_alloc_state()` — e.g. "40% vol = fully stressed". These are market-specific.
4. **Beta window** for ETF engine — longer = more stable betas but slower to adapt.
5. **`recovery_factor`** — how much of the drawdown must recover before re-engaging.

---

## Dependencies

- `numpy` — numerics everywhere
- `scipy` — SLSQP quadratic programme solver
- `pyyaml` — config loading
- `pytest` — tests

No external LP solver, no heavy ML dependency required to run the allocator and stubs.
