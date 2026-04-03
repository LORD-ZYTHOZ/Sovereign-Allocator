"""
qp_solver.py — Quadratic Programme solver for strategy budget allocation

Solves:
    b* = argmax_b  [ ν^T b  −  (γ/2) b^T Σ b  −  λ_turn ‖b − b_{t-1}‖_1 ]

    subject to:
        1^T b ≤ 1          (total budget ≤ 100% of capital)
        b ≥ 0              (long-only budgets, no negative allocation to an engine)

The ‖b − b_{t-1}‖_1 turnover penalty is linearised via auxiliary variables:
    introduce  δ+ = max(b − b_prev, 0),  δ- = max(b_prev − b, 0)
    so ‖b − b_prev‖_1 = 1^T (δ+ + δ-)

This converts the problem to a standard QP with linear inequality constraints,
solvable with scipy.optimize.minimize (SLSQP) — no external LP/QP solver needed.

Adaptive γ
----------
γ_t is scaled by the volatility regime: higher vol → larger γ → more risk aversion.
    γ_t = γ_base × (1 + vol_regime_scale × volatility_regime)

Adaptive λ_turn
--------------
λ_turn is scaled by the liquidity score: illiquid markets → higher turnover penalty.
    λ_t = λ_base × (1 + liq_penalty × (1 − liquidity_score))
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds


class QPAllocator:
    """
    Parameters
    ----------
    engine_ids    : ordered list of engine IDs (must match Σ ordering)
    gamma_base    : base risk-aversion coefficient
    lambda_turn   : base turnover penalty
    vol_regime_scale  : how much γ scales with volatility regime [0,1]
    liq_penalty       : how much λ_turn scales with illiquidity [0,1]
    budget_cap        : max fraction any single engine can receive [0,1]
    """

    def __init__(
        self,
        engine_ids: List[str],
        gamma_base: float = 2.0,
        lambda_turn: float = 0.1,
        vol_regime_scale: float = 1.5,
        liq_penalty: float = 0.5,
        budget_cap: float = 0.80,
    ):
        self.engine_ids       = engine_ids
        self.n                = len(engine_ids)
        self.gamma_base       = gamma_base
        self.lambda_turn      = lambda_turn
        self.vol_regime_scale = vol_regime_scale
        self.liq_penalty      = liq_penalty
        self.budget_cap       = budget_cap

        self._prev_b = np.ones(self.n) / self.n   # equal-weight start

    def solve(
        self,
        utility_scores: Dict[str, float],   # ν_t^s
        sigma: np.ndarray,                   # Σ_t^{slv}  (n×n)
        volatility_regime: float = 0.5,
        liquidity_score: float = 0.5,
    ) -> Tuple[Dict[str, float], float, str]:
        """
        Returns
        -------
        budgets       : {engine_id: budget fraction}
        cash_fraction : 1 − sum(budgets)
        status        : "optimal" | "fallback"
        """
        nu = np.array([utility_scores.get(e, 0.0) for e in self.engine_ids])

        # Adaptive parameters
        gamma_t  = self.gamma_base * (1 + self.vol_regime_scale * volatility_regime)
        lambda_t = self.lambda_turn * (1 + self.liq_penalty * (1 - liquidity_score))

        b_prev = self._prev_b.copy()
        n = self.n

        # ── Objective (negated for minimisation) ──────────────────────────
        # f(b) = −ν^T b  +  (γ/2) b^T Σ b  +  λ ‖b − b_prev‖_1
        # Linearise L1: introduce slack variables d = [d+, d-]  size 2n
        # Variables: x = [b (n), d+ (n), d- (n)]   total: 3n

        def objective(x: np.ndarray) -> float:
            b   = x[:n]
            dp  = x[n:2*n]
            dm  = x[2*n:]
            quad = 0.5 * gamma_t * b @ sigma @ b
            return -float(nu @ b) + quad + lambda_t * (dp + dm).sum()

        def grad(x: np.ndarray) -> np.ndarray:
            b  = x[:n]
            dp = x[n:2*n]
            dm = x[2*n:]
            gb  = -nu + gamma_t * (sigma @ b)
            gdp = np.full(n, lambda_t)
            gdm = np.full(n, lambda_t)
            return np.concatenate([gb, gdp, gdm])

        # ── Constraints ───────────────────────────────────────────────────
        # 1)  sum(b) ≤ 1
        # 2)  b_i ≤ budget_cap
        # 3)  b − b_prev ≤ d+  →  b - d+ ≤ b_prev
        # 4)  b_prev − b ≤ d-  →  -b - d- ≤ -b_prev
        # 5)  d+, d- ≥ 0
        # 6)  b ≥ 0

        constraints = []

        # budget sum ≤ 1
        A_sum = np.zeros((1, 3*n))
        A_sum[0, :n] = 1.0
        constraints.append(LinearConstraint(A_sum, lb=-np.inf, ub=1.0))

        # |b − b_prev| via d+, d-
        # b - d+ ≤ b_prev  →  Ax ≤ b_prev
        A_dp = np.zeros((n, 3*n))
        A_dp[:, :n]  =  np.eye(n)
        A_dp[:, n:2*n] = -np.eye(n)
        constraints.append(LinearConstraint(A_dp, lb=-np.inf, ub=b_prev))

        # -b - d- ≤ -b_prev
        A_dm = np.zeros((n, 3*n))
        A_dm[:, :n]   = -np.eye(n)
        A_dm[:, 2*n:] = -np.eye(n)
        constraints.append(LinearConstraint(A_dm, lb=-np.inf, ub=-b_prev))

        # bounds: b ∈ [0, budget_cap], d+/d- ∈ [0, ∞)
        lb = np.zeros(3*n)
        ub = np.concatenate([
            np.full(n, self.budget_cap),
            np.full(2*n, np.inf),
        ])
        bounds = Bounds(lb=lb, ub=ub)

        # initial guess
        x0 = np.concatenate([b_prev, np.zeros(2*n)])

        result = minimize(
            fun=objective,
            jac=grad,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {"type": "ineq", "fun": lambda x: 1.0 - x[:n].sum(), "jac": lambda x: np.concatenate([-np.ones(n), np.zeros(2*n)])},
                {"type": "ineq", "fun": lambda x: x[n:2*n] - (x[:n] - b_prev), "jac": lambda x: np.concatenate([-np.eye(n), np.eye(n), np.zeros((n, n))], axis=1)},
                {"type": "ineq", "fun": lambda x: x[2*n:] - (b_prev - x[:n]), "jac": lambda x: np.concatenate([np.eye(n), np.zeros((n, n)), np.eye(n)], axis=1)},
            ],
            options={"maxiter": 200, "ftol": 1e-9},
        )

        if result.success:
            b_star = np.clip(result.x[:n], 0.0, self.budget_cap)
            status = "optimal"
        else:
            # Fallback: proportional to utility (softmax), clipped
            pos_nu = np.clip(nu, 0, None)
            if pos_nu.sum() > 1e-8:
                b_star = pos_nu / pos_nu.sum() * 0.8
            else:
                b_star = np.ones(n) / n * 0.5
            status = "fallback"

        # Normalise: ensure sum ≤ 1
        if b_star.sum() > 1.0:
            b_star /= b_star.sum()

        self._prev_b = b_star.copy()

        budgets       = {e: float(b_star[i]) for i, e in enumerate(self.engine_ids)}
        cash_fraction = float(1.0 - b_star.sum())

        return budgets, cash_fraction, status
