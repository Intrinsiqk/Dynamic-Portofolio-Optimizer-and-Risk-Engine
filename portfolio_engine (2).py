"""
Dynamic Portfolio Optimizer & Risk Engine
=========================================

Components
----------
1. Data            : regime-switching market simulator (offline demo), CSV / yfinance loaders
2. Estimators      : sample, EWMA and Ledoit-Wolf covariance; shrunk expected returns
3. Optimizers      : min-variance, max-Sharpe, mean-variance, risk parity, min-CVaR (LP)
4. Risk engine     : historical / parametric / Cornish-Fisher / Monte-Carlo VaR & CVaR,
                     risk contributions, drawdowns, stress tests, Kupiec VaR backtest
5. Dynamic layer   : walk-forward backtester with periodic rebalancing, transaction costs,
                     volatility targeting and a drawdown circuit breaker

Run the demo:
    python portfolio_engine.py                       # simulated market
    python portfolio_engine.py --csv prices.csv      # your own daily prices (Date index)
    python portfolio_engine.py --tickers SPY TLT GLD --start 2015-01-01   # needs `pip install yfinance`

Educational software: not investment advice.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.optimize import linprog, minimize

TRADING_DAYS = 252


# =============================================================================
# 1. DATA
# =============================================================================
ASSETS = ["SPY", "EFA", "EEM", "VNQ", "TLT", "IEF", "GLD", "DBC"]
_CALM_MU = [0.10, 0.07, 0.08, 0.08, 0.03, 0.025, 0.05, 0.04]
_CALM_VOL = [0.14, 0.16, 0.20, 0.19, 0.13, 0.06, 0.15, 0.18]
_CRISIS_MU = [-0.25, -0.30, -0.40, -0.35, 0.10, 0.06, 0.10, -0.20]
_CRISIS_VOL = [0.35, 0.38, 0.45, 0.42, 0.16, 0.08, 0.22, 0.30]


def _block_corr(eq_eq, bond_bond, eq_bond, eq_gold, eq_comm, bond_gold, bond_comm, gold_comm):
    """Correlation matrix from asset-class blocks: equities(0-3), bonds(4-5), gold(6), commodities(7)."""
    n = 8
    C = np.eye(n)
    groups = {"eq": [0, 1, 2, 3], "bond": [4, 5], "gold": [6], "comm": [7]}
    pairs = {("eq", "eq"): eq_eq, ("bond", "bond"): bond_bond, ("eq", "bond"): eq_bond,
             ("eq", "gold"): eq_gold, ("eq", "comm"): eq_comm, ("bond", "gold"): bond_gold,
             ("bond", "comm"): bond_comm, ("gold", "comm"): gold_comm}
    for (a, b), rho in pairs.items():
        for i in groups[a]:
            for j in groups[b]:
                if i != j:
                    C[i, j] = C[j, i] = rho
    assert np.linalg.eigvalsh(C).min() > 0, "correlation matrix not positive definite"
    return C


def simulate_market(n_days: int = 2520, seed: int = 42, df: float = 5.0) -> pd.DataFrame:
    """
    Two-regime (calm / crisis) Markov-switching market with fat-tailed (Student-t) shocks.
    In crisis: vols jump, equity correlations rise, bonds and gold become diversifiers.
    Returns daily *simple* returns.
    """
    rng = np.random.default_rng(seed)
    p_calm_to_crisis, p_crisis_to_calm = 0.006, 0.04

    corr = {
        0: _block_corr(0.70, 0.85, -0.20, 0.05, 0.40, 0.25, -0.05, 0.30),
        1: _block_corr(0.90, 0.85, -0.45, 0.00, 0.60, 0.30, -0.20, 0.30),
    }
    chol = {k: np.linalg.cholesky(v) for k, v in corr.items()}
    mu = {0: np.array(_CALM_MU) / TRADING_DAYS, 1: np.array(_CRISIS_MU) / TRADING_DAYS}
    sig = {0: np.array(_CALM_VOL) / np.sqrt(TRADING_DAYS), 1: np.array(_CRISIS_VOL) / np.sqrt(TRADING_DAYS)}

    regime, out = 0, np.empty((n_days, len(ASSETS)))
    regimes = np.empty(n_days, dtype=int)
    for t in range(n_days):
        u = rng.random()
        if regime == 0 and u < p_calm_to_crisis:
            regime = 1
        elif regime == 1 and u < p_crisis_to_calm:
            regime = 0
        regimes[t] = regime
        # unit-variance Student-t innovations
        z = rng.standard_normal(len(ASSETS))
        chi = rng.chisquare(df) / df
        z = z / np.sqrt(chi) * np.sqrt((df - 2) / df)
        out[t] = mu[regime] + sig[regime] * (chol[regime] @ z)

    idx = pd.bdate_range(start="2016-01-04", periods=n_days)
    df_out = pd.DataFrame(np.clip(out, -0.99, None), index=idx, columns=ASSETS)
    df_out.attrs["regimes"] = pd.Series(regimes, index=idx, name="regime")
    return df_out


def load_returns_from_csv(path: str) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return prices.pct_change().dropna(how="all").dropna()


def load_returns_yfinance(tickers, start="2010-01-01", end=None) -> pd.DataFrame:
    import yfinance as yf  # optional dependency
    px = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return px.pct_change().dropna()


# =============================================================================
# 2. ESTIMATORS  (all return DAILY moments; the backtester annualizes)
# =============================================================================
def sample_cov(X: np.ndarray) -> np.ndarray:
    return np.cov(X, rowvar=False)


def ewma_cov(X: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics-style exponentially weighted covariance (zero-mean convention)."""
    T = len(X)
    w = lam ** np.arange(T - 1, -1, -1)
    w /= w.sum()
    return X.T @ (w[:, None] * X)


def ledoit_wolf_cov(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) shrinkage toward a scaled identity matrix."""
    X = X - X.mean(axis=0)
    n, p = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / p
    d2 = np.sum((S - mu * np.eye(p)) ** 2)
    sq_norms = np.sum(X * X, axis=1)                       # x_k'x_k
    b2 = (np.sum(sq_norms ** 2) - 2 * np.sum((X @ S) * X) + n * np.sum(S ** 2)) / n ** 2
    b2 = min(b2, d2)
    shrink = 0.0 if d2 == 0 else b2 / d2
    return shrink * mu * np.eye(p) + (1 - shrink) * S


COV_ESTIMATORS = {"sample": sample_cov, "ewma": ewma_cov, "ledoit_wolf": ledoit_wolf_cov}


def estimate_mu(X: np.ndarray, shrink: float = 0.5) -> np.ndarray:
    """Annualized mean returns, shrunk toward the cross-sectional mean (sample means are very noisy)."""
    m = X.mean(axis=0) * TRADING_DAYS
    return (1 - shrink) * m + shrink * m.mean()


# =============================================================================
# 3. OPTIMIZERS  (inputs are ANNUALIZED mu / cov)
# =============================================================================
class PortfolioOptimizer:
    def __init__(self, mu, cov, rf: float = 0.0, max_weight: float = 1.0, min_weight: float = 0.0):
        self.mu = np.asarray(mu, float)
        self.cov = np.asarray(cov, float)
        self.rf = rf
        self.n = len(self.mu)
        if max_weight * self.n < 1 - 1e-9:
            raise ValueError("max_weight too small: weights cannot sum to 1")
        self.lo, self.hi = min_weight, max_weight
        self.bounds = [(min_weight, max_weight)] * self.n
        self.constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # ---- helpers -----------------------------------------------------------
    def _solve(self, fun, x0=None) -> np.ndarray:
        x0 = np.full(self.n, 1.0 / self.n) if x0 is None else x0
        res = minimize(fun, x0, method="SLSQP", bounds=self.bounds, constraints=self.constraints,
                       options={"ftol": 1e-12, "maxiter": 500})
        if not res.success:
            warnings.warn(f"optimizer did not fully converge: {res.message}")
        w = np.clip(res.x, self.lo, self.hi)
        return w / w.sum()

    def variance(self, w):
        return float(w @ self.cov @ w)

    def risk_contributions(self, w) -> np.ndarray:
        """Fraction of total portfolio variance contributed by each asset (sums to 1)."""
        return w * (self.cov @ w) / self.variance(w)

    # ---- strategies --------------------------------------------------------
    def min_variance(self) -> np.ndarray:
        return self._solve(lambda w: w @ self.cov @ w)

    def max_sharpe(self) -> np.ndarray:
        return self._solve(lambda w: -(w @ self.mu - self.rf) / np.sqrt(w @ self.cov @ w))

    def mean_variance(self, risk_aversion: float = 5.0) -> np.ndarray:
        return self._solve(lambda w: 0.5 * risk_aversion * (w @ self.cov @ w) - w @ self.mu)

    def risk_parity(self) -> np.ndarray:
        """
        Equal risk contribution via Spinu's (2013) convex reformulation:

            min_y  0.5 * y' Σ y  -  c * Σ_i ln(y_i),      y > 0

        The first-order condition is y_i (Σy)_i = c for every i, i.e. all assets contribute equally
        to risk. The Hessian Σ + c*diag(1/y²) is positive definite, so the problem is strictly convex
        with a unique minimizer. Rescaling y to sum to 1 preserves the risk-contribution ratios.
        """
        n, cov = self.n, self.cov
        inv_vol = 1.0 / np.sqrt(np.diag(cov))
        x0 = inv_vol / inv_vol.sum()
        c = float(x0 @ cov @ x0) / n            # any c > 0 works; this keeps sum(y) near 1 (conditioning only)

        fun = lambda y: 0.5 * y @ cov @ y - c * np.sum(np.log(y))
        jac = lambda y: cov @ y - c / y
        hess = lambda y: cov + np.diag(c / y ** 2)

        res = minimize(fun, x0, jac=jac, hess=hess, method="trust-constr",
                       bounds=[(1e-10, None)] * n, options={"gtol": 1e-12, "xtol": 1e-12, "maxiter": 1000})
        if not res.success:
            warnings.warn(f"risk_parity did not fully converge: {res.message}")
        w = res.x / res.x.sum()

        # The pure ERC solution ignores box constraints. If a cap/floor binds, project back
        # (water-filling); exact equal risk is then unattainable, so warn.
        if w.max() > self.hi + 1e-9 or w.min() < self.lo - 1e-9:
            warnings.warn("risk_parity: weight bounds bind; returning bounded approximation of equal risk contribution")
            for _ in range(n):
                w = np.clip(w, self.lo, self.hi)
                gap = 1.0 - w.sum()
                if abs(gap) < 1e-12:
                    break
                free = (w < self.hi - 1e-12) if gap > 0 else (w > self.lo + 1e-12)
                if not free.any():
                    break
                w[free] += gap * w[free] / w[free].sum()
        return w

    def min_cvar(self, scenarios: np.ndarray, beta: float = 0.95, min_return: float | None = None) -> np.ndarray:
        """
        Rockafellar-Uryasev linear program: minimize CVaR_beta of historical scenario returns.
        `scenarios` is a (T x n) array of daily returns; `min_return` is a DAILY expected-return floor.

        Failure handling:
          * weight bounds that cannot sum to 1 -> ValueError (no fallback can fix that);
          * infeasible / failed LP with a return floor -> warn, relax (drop) the floor and re-solve;
          * LP still failing -> warn and fall back to min_variance().
        """
        scenarios = np.asarray(scenarios, float)
        T, n = scenarios.shape
        if n * self.lo > 1 + 1e-9 or n * self.hi < 1 - 1e-9:
            raise ValueError(f"weight bounds [{self.lo}, {self.hi}] cannot satisfy sum(w) = 1 for {n} assets")

        def _solve(floor):
            c = np.r_[np.zeros(n), 1.0, np.full(T, 1.0 / ((1 - beta) * T))]      # [w, alpha, u]
            A_ub = sparse.hstack([sparse.csr_matrix(-scenarios), sparse.csr_matrix(-np.ones((T, 1))),
                                  -sparse.identity(T)]).tocsr()                  # -r.w - alpha - u <= 0
            b_ub = np.zeros(T)
            if floor is not None:
                row = sparse.csr_matrix(np.r_[-scenarios.mean(axis=0), 0.0, np.zeros(T)][None, :])
                A_ub, b_ub = sparse.vstack([A_ub, row]).tocsr(), np.r_[b_ub, -floor]
            A_eq = sparse.csr_matrix(np.r_[np.ones(n), 0.0, np.zeros(T)][None, :])
            bounds = self.bounds + [(None, None)] + [(0, None)] * T
            return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")

        res = _solve(min_return)
        if not res.success and min_return is not None:
            print(f"[min_cvar] WARNING: LP failed with min_return={min_return:.3e} "
                  f"(status {res.status}: {res.message}). Relaxing the return floor and re-solving.")
            res = _solve(None)
        if not res.success:
            print(f"[min_cvar] WARNING: LP failed (status {res.status}: {res.message}). "
                  "Falling back to min_variance().")
            return self.min_variance()

        w = np.clip(res.x[:n], self.lo, self.hi)
        return w / w.sum()

    def efficient_frontier(self, n_points: int = 25) -> pd.DataFrame:
        """Long-only frontier via target-return constrained minimum variance."""
        rows = []
        for tr in np.linspace(self.mu.min(), self.mu.max(), n_points):
            cons = self.constraints + [{"type": "eq", "fun": lambda w, tr=tr: w @ self.mu - tr}]
            res = minimize(lambda w: w @ self.cov @ w, np.full(self.n, 1 / self.n), method="SLSQP",
                           bounds=self.bounds, constraints=cons, options={"ftol": 1e-12, "maxiter": 500})
            if res.success:
                rows.append({"return": tr, "vol": np.sqrt(res.fun)})
        return pd.DataFrame(rows)


# =============================================================================
# 4. RISK ENGINE
# =============================================================================
class RiskEngine:
    """
    Risk analytics for a fixed weight vector. `weights` is a Series indexed by asset name;
    it may sum to less than 1 (the remainder is treated as cash with zero return).
    """

    def __init__(self, returns: pd.DataFrame, weights: pd.Series):
        self.returns = returns[weights.index]
        self.weights = weights
        self.port = self.returns @ weights                      # daily portfolio returns
        self.cov = self.returns.cov().values                    # daily covariance

    # ---- VaR / CVaR (reported as positive loss fractions) ------------------
    def historical_var(self, alpha=0.95) -> float:
        return float(-np.quantile(self.port, 1 - alpha))

    def historical_cvar(self, alpha=0.95) -> float:
        q = np.quantile(self.port, 1 - alpha)
        return float(-self.port[self.port <= q].mean())

    def parametric_var(self, alpha=0.95) -> float:
        return float(-(self.port.mean() + stats.norm.ppf(1 - alpha) * self.port.std()))

    def cornish_fisher_var(self, alpha=0.95) -> float:
        """Normal VaR adjusted for skewness and excess kurtosis."""
        z = stats.norm.ppf(1 - alpha)
        s, k = stats.skew(self.port), stats.kurtosis(self.port)
        zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
        return float(-(self.port.mean() + zcf * self.port.std()))

    def monte_carlo_var(self, alpha=0.95, n_sims=200_000, df=5, seed=0, horizon=1) -> tuple[float, float]:
        """
        VaR & CVaR from a multivariate Student-t simulation matching the sample mean/covariance.
        Over `horizon` days the drift grows linearly (mu * h) and the shocks with the square root
        of time (z * sqrt(h)); scaling the *sum* by sqrt(h) would wrongly shrink/inflate the drift.
        """
        rng = np.random.default_rng(seed)
        mu = self.returns.mean().values
        L = np.linalg.cholesky(self.cov + 1e-12 * np.eye(len(mu)))
        z = rng.standard_normal((n_sims, len(mu))) @ L.T
        z *= np.sqrt((df - 2) / df) * np.sqrt(df / rng.chisquare(df, n_sims))[:, None]
        sim = (mu * horizon + z * np.sqrt(horizon)) @ self.weights.values
        q = np.quantile(sim, 1 - alpha)
        return float(-q), float(-sim[sim <= q].mean())

    # ---- structure ---------------------------------------------------------
    def risk_contributions(self) -> pd.DataFrame:
        w = self.weights.values
        port_vol = np.sqrt(w @ self.cov @ w)
        marginal = self.cov @ w / port_vol
        comp = w * marginal
        return pd.DataFrame({"weight": w, "marginal_risk": marginal * np.sqrt(TRADING_DAYS),
                             "component_risk": comp * np.sqrt(TRADING_DAYS),
                             "pct_of_risk": comp / port_vol}, index=self.weights.index)

    def diversification_ratio(self) -> float:
        w, vols = self.weights.values, np.sqrt(np.diag(self.cov))
        return float(w @ vols / np.sqrt(w @ self.cov @ w))

    def max_drawdown(self) -> float:
        return max_drawdown(self.port)

    def stress_test(self, scenarios: dict[str, dict[str, float]]) -> pd.Series:
        """Instantaneous P&L (fraction of NAV) for user-defined asset shocks, e.g. {'SPY': -0.30}."""
        out = {}
        for name, shocks in scenarios.items():
            s = pd.Series(shocks).reindex(self.weights.index).fillna(0.0)
            out[name] = float((self.weights * s).sum())
        return pd.Series(out, name="pnl")

    def report(self, alpha=0.95) -> pd.Series:
        mc_var, mc_cvar = self.monte_carlo_var(alpha)
        return pd.Series({
            f"Hist VaR {alpha:.0%} (1d)": self.historical_var(alpha),
            f"Hist CVaR {alpha:.0%} (1d)": self.historical_cvar(alpha),
            f"Normal VaR {alpha:.0%} (1d)": self.parametric_var(alpha),
            f"Cornish-Fisher VaR {alpha:.0%} (1d)": self.cornish_fisher_var(alpha),
            f"Monte-Carlo t VaR {alpha:.0%} (1d)": mc_var,
            f"Monte-Carlo t CVaR {alpha:.0%} (1d)": mc_cvar,
            "Ann. volatility": float(self.port.std() * np.sqrt(TRADING_DAYS)),
            "Max drawdown": self.max_drawdown(),
            "Diversification ratio": self.diversification_ratio(),
        })


DEFAULT_STRESS = {
    "Equity crash (-30%)": {"SPY": -0.30, "EFA": -0.33, "EEM": -0.40, "VNQ": -0.35, "TLT": 0.08, "IEF": 0.04, "GLD": 0.05, "DBC": -0.15},
    "Rates shock (+200bp)": {"SPY": -0.08, "EFA": -0.07, "EEM": -0.09, "VNQ": -0.15, "TLT": -0.18, "IEF": -0.08, "GLD": -0.05, "DBC": 0.02},
    "Stagflation": {"SPY": -0.15, "EFA": -0.15, "EEM": -0.12, "VNQ": -0.15, "TLT": -0.12, "IEF": -0.06, "GLD": 0.15, "DBC": 0.25},
    "Correlation-1 selloff": {a: -0.15 for a in ASSETS},
}


def max_drawdown(returns: pd.Series) -> float:
    eq = (1 + returns).cumprod()
    return float((eq / eq.cummax() - 1).min())


def kupiec_pof(exceptions: int, n: int, p: float) -> float:
    """Kupiec proportion-of-failures test. Returns the p-value (small => VaR model rejected)."""
    x = exceptions
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)) + 2 * ((n - x) * np.log(1 - x / n) + x * np.log(x / n))
    return float(1 - stats.chi2.cdf(lr, 1))


def var_backtest(returns: pd.Series, alpha=0.95, window=250) -> dict:
    """Rolling one-day historical VaR vs realized returns (strictly causal)."""
    # Lag FIRST, then roll: the threshold for day t is built only from days t-window .. t-1,
    # so r_t can never influence the VaR it is tested against.
    thresh = returns.shift(1).rolling(window).quantile(1 - alpha).dropna()
    realized = returns.loc[thresh.index]
    exc = int((realized < thresh).sum())
    n = len(realized)
    return {"observations": n, "exceptions": exc, "expected": round(n * (1 - alpha), 1),
            "exception_rate": exc / n, "kupiec_pvalue": kupiec_pof(exc, n, 1 - alpha)}


def performance_summary(r: pd.Series, rf: float = 0.0) -> pd.Series:
    r = r.dropna()
    years = len(r) / TRADING_DAYS
    cagr = (1 + r).prod() ** (1 / years) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    excess = r.mean() * TRADING_DAYS - rf
    downside = np.sqrt(np.mean(np.minimum(r - rf / TRADING_DAYS, 0) ** 2)) * np.sqrt(TRADING_DAYS)
    mdd = max_drawdown(r)
    q = np.quantile(r, 0.05)
    return pd.Series({"CAGR": cagr, "Volatility": vol, "Sharpe": excess / vol, "Sortino": excess / downside,
                      "Max drawdown": mdd, "Calmar": cagr / abs(mdd) if mdd else np.nan,
                      "VaR 95% (1d)": -q, "CVaR 95% (1d)": -r[r <= q].mean()})


# =============================================================================
# 5. DYNAMIC LAYER: walk-forward backtester
# =============================================================================
@dataclass
class BacktestConfig:
    method: str = "risk_parity"       # equal_weight | min_variance | max_sharpe | mean_variance | risk_parity | min_cvar
    lookback: int = 504               # trading days of history used at each rebalance
    rebalance_every: int = 21         # scheduled rebalance frequency (days)
    max_weight: float = 0.40
    cov_method: str = "ledoit_wolf"   # sample | ewma | ledoit_wolf
    mu_shrink: float = 0.5            # 0 = raw sample means, 1 = all assets get the same expected return
    risk_aversion: float = 5.0        # only for mean_variance
    cost_bps: float = 5.0             # one-way transaction cost, in bps of traded notional
    rf: float = 0.0                   # annual risk-free rate (also the cash return)
    vol_target: float | None = 0.10   # annualized vol target for the overlay (None = off)
    max_leverage: float = 1.0         # cap on overlay exposure (1.0 = never borrow)
    dd_limit: float | None = 0.10     # drawdown circuit breaker (None = off)
    dd_derisk: float = 0.5            # exposure multiplier while breaker is on
    breaker_cooldown: int = 63        # days before the breaker resets even without recovery


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series
    weights: pd.DataFrame             # target weights at each rebalance (incl. CASH)
    turnover: pd.Series
    exposure: pd.Series               # risky-asset exposure at each rebalance
    breaker_dates: list = field(default_factory=list)


class Backtester:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg

    def _risky_weights(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        n = window.shape[1]
        cov_ann = COV_ESTIMATORS[c.cov_method](window) * TRADING_DAYS
        if c.method == "equal_weight":
            return np.full(n, 1.0 / n), cov_ann
        mu = estimate_mu(window, c.mu_shrink)
        opt = PortfolioOptimizer(mu, cov_ann, c.rf, c.max_weight)
        if c.method == "min_variance":
            w = opt.min_variance()
        elif c.method == "max_sharpe":
            w = opt.max_sharpe()
        elif c.method == "mean_variance":
            w = opt.mean_variance(c.risk_aversion)
        elif c.method == "risk_parity":
            w = opt.risk_parity()
        elif c.method == "min_cvar":
            w = opt.min_cvar(window)
        else:
            raise ValueError(f"unknown method {c.method}")
        return w, cov_ann

    def run(self, returns: pd.DataFrame, name: str | None = None) -> BacktestResult:
        c = self.cfg
        R, dates = returns.values, returns.index
        T, n = R.shape
        rf_d = c.rf / TRADING_DAYS

        w = np.zeros(n + 1)
        w[-1] = 1.0                                      # start 100% cash
        equity = peak = 1.0
        breaker, breaker_since = False, 0
        rec = {"dates": [], "w": [], "exp": []}
        port_ret, turns, breaker_dates = [], [], []

        for i in range(c.lookback, T):
            # ---- circuit-breaker state machine (with hysteresis + cooldown) ----
            dd = equity / peak - 1
            new_breaker = breaker
            if c.dd_limit is not None:
                if not breaker and dd < -c.dd_limit:
                    new_breaker, breaker_since = True, i
                elif breaker and (dd > -c.dd_limit / 2 or i - breaker_since >= c.breaker_cooldown):
                    new_breaker = False
                    peak = equity                          # reset high-water mark after re-risking
            scheduled = (i - c.lookback) % c.rebalance_every == 0
            cost = turnover = 0.0

            if scheduled or new_breaker != breaker:
                if new_breaker and not breaker:
                    breaker_dates.append(dates[i])
                breaker = new_breaker
                window = R[i - c.lookback:i]
                w_risky, cov_ann = self._risky_weights(window)

                exposure = 1.0
                if c.vol_target is not None:
                    vol_fc = np.sqrt(w_risky @ (ewma_cov(window) * TRADING_DAYS) @ w_risky)
                    exposure = float(np.clip(c.vol_target / vol_fc, 0.0, c.max_leverage))
                if breaker:
                    exposure *= c.dd_derisk

                target = np.r_[w_risky * exposure, 1.0 - exposure]
                turnover = float(np.abs(target[:n] - w[:n]).sum())      # cash leg trades for free
                cost = turnover * c.cost_bps / 1e4
                w = target
                rec["dates"].append(dates[i]); rec["w"].append(target.copy()); rec["exp"].append(exposure)

            r_all = np.r_[R[i], rf_d]
            gross = float(w @ r_all)
            net = gross - cost
            equity *= 1 + net
            peak = max(peak, equity)
            port_ret.append(net); turns.append(turnover)
            w = w * (1 + r_all) / (1 + gross)               # weights drift between rebalances

        idx = dates[c.lookback:]
        cols = list(returns.columns) + ["CASH"]
        return BacktestResult(
            name=name or c.method,
            returns=pd.Series(port_ret, index=idx, name=name or c.method),
            weights=pd.DataFrame(rec["w"], index=rec["dates"], columns=cols),
            turnover=pd.Series(turns, index=idx),
            exposure=pd.Series(rec["exp"], index=rec["dates"]),
            breaker_dates=breaker_dates,
        )


# =============================================================================
# REPORTING / DEMO
# =============================================================================
def plot_report(results: dict[str, BacktestResult], featured: str, engine: RiskEngine, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    for name, res in results.items():
        eq = (1 + res.returns).cumprod()
        lw = 2.4 if name == featured else 1.1
        ax[0, 0].plot(eq.index, eq, label=name, lw=lw)
    ax[0, 0].set_title("Growth of $1 (net of costs)"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)

    for name in [featured, "Equal weight (no overlay)"]:
        if name in results:
            eq = (1 + results[name].returns).cumprod()
            ax[0, 1].fill_between(eq.index, eq / eq.cummax() - 1, 0, alpha=.4, label=name)
    for d in results[featured].breaker_dates:
        ax[0, 1].axvline(d, color="red", alpha=.4, lw=.8)
    ax[0, 1].set_title("Drawdowns (red lines = circuit breaker trips)"); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.3)

    W = results[featured].weights
    ax[1, 0].stackplot(W.index, W.T.values, labels=W.columns)
    ax[1, 0].set_title(f"Target weights: {featured}"); ax[1, 0].legend(fontsize=7, ncol=3, loc="lower left")

    rc = engine.risk_contributions()
    x = np.arange(len(rc))
    ax[1, 1].bar(x - 0.2, rc["weight"], 0.4, label="Weight")
    ax[1, 1].bar(x + 0.2, rc["pct_of_risk"], 0.4, label="% of risk")
    ax[1, 1].set_xticks(x, rc.index); ax[1, 1].set_title("Latest allocation: capital vs risk"); ax[1, 1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Dynamic Portfolio Optimizer & Risk Engine demo")
    ap.add_argument("--csv", help="CSV of daily prices (first column = date)")
    ap.add_argument("--tickers", nargs="+", help="tickers to download via yfinance")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--out", default="portfolio_report.png")
    args = ap.parse_args()

    if args.csv:
        rets = load_returns_from_csv(args.csv)
    elif args.tickers:
        rets = load_returns_yfinance(args.tickers, args.start)
    else:
        rets = simulate_market()
        print("Using simulated regime-switching market data.\n")
    print(f"{rets.shape[1]} assets, {len(rets)} days: {rets.index[0].date()} -> {rets.index[-1].date()}\n")

    base = dict(lookback=504, rebalance_every=21, max_weight=0.40, dd_limit=0.05)
    off = dict(vol_target=None, dd_limit=None)
    configs = {
        "Equal weight (no overlay)": BacktestConfig(method="equal_weight", **{**base, **off}),
        "Equal weight + overlay": BacktestConfig(method="equal_weight", **base),
        "Min variance": BacktestConfig(method="min_variance", **base),
        "Max Sharpe": BacktestConfig(method="max_sharpe", **base),
        "Risk parity": BacktestConfig(method="risk_parity", **base),
        "Min CVaR": BacktestConfig(method="min_cvar", **base),
        "Risk parity (no overlay)": BacktestConfig(method="risk_parity", **{**base, **off}),
    }
    results = {name: Backtester(cfg).run(rets, name) for name, cfg in configs.items()}

    table = pd.DataFrame({n: performance_summary(r.returns) for n, r in results.items()}).T
    table["Avg turnover/rebal"] = [r.turnover[r.turnover > 0].mean() for r in results.values()]
    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.width = 250
    pd.options.display.max_columns = 20
    print("=== Walk-forward backtest (net of 5 bps costs) ===")
    print(table, "\n")

    featured = "Risk parity"
    last_w = results[featured].weights.iloc[-1].drop("CASH")
    engine = RiskEngine(rets.iloc[-504:], last_w)
    print(f"=== Risk report: latest '{featured}' allocation (cash = {1 - last_w.sum():.1%}) ===")
    print(engine.report(), "\n")
    print(engine.risk_contributions(), "\n")
    if set(ASSETS) >= set(rets.columns):
        print("=== Stress tests (P&L as % of NAV) ===")
        print(engine.stress_test(DEFAULT_STRESS), "\n")
    bt = var_backtest(results[featured].returns)
    print("=== 95% VaR backtest (rolling 250d historical VaR) ===")
    print(bt, "\n")

    plot_report(results, featured, engine, args.out)
    print(f"Chart saved to {args.out}")


if __name__ == "__main__":
    main()
