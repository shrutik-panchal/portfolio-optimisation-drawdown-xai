#-------------------------------------------------------------------------------------
"""
Mean-Variance Portfolio Optimisation (Markowitz 1952).

Provides three core optimisation targets:
  1. Maximum Sharpe Ratio   — best risk-adjusted return
  2. Minimum Volatility     — lowest portfolio risk
  3. Efficient Frontier     — N portfolios tracing the optimal risk/return curve

All functions integrate directly with data/fetch_data.py via the (prices, returns) tuple returned by get_data().

Usage (standalone):
    from data.fetch_data import get_data
    from models.markowitz import run_markowitz

    prices, _ = get_data()
    results   = run_markowitz(prices)
    print(results["max_sharpe"]["weights"])
"""
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Importing required libraries
#-------------------------------------------------------------------------------------
import logging
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.exceptions import OptimizationError
#-------------------------------------------------------------------------------------
log = logging.getLogger(__name__)
#-------------------------------------------------------------------------------------
TRADING_DAYS   = 252
RISK_FREE_RATE = 0.0525   # RBI repo rate approximation as per April 2026 (update as needed)
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 1: Input Preparation 
#-------------------------------------------------------------------------------------
def compute_expected_returns(prices, method="mean_historical_return"):
    if method == "ema_historical_return":
        mu = expected_returns.ema_historical_return(prices, frequency=TRADING_DAYS)
    else:
        mu = expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)
    log.info(f"Expected returns ({method}):\n{mu.round(4)}")
    return mu
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Compute covariance matrix using specified method. Ledoit-Wolf shrinkage is preferred
#-------------------------------------------------------------------------------------
def compute_covariance(prices, method="ledoit_wolf"):
    """
    Ledoit-Wolf shrinkage is preferred over raw sample covariance.
    Sample covariance is noisy for small T/N ratios — common in equity
    portfolios with fewer than 5 years of data.
    """
    if method == "exp_cov":
        S = risk_models.exp_cov(prices, frequency=TRADING_DAYS)
    elif method == "sample_cov":
        S = risk_models.sample_cov(prices, frequency=TRADING_DAYS)
    else:
        S = risk_models.CovarianceShrinkage(prices, frequency=TRADING_DAYS).ledoit_wolf()
    log.info(f"Covariance matrix ({method}), shape: {S.shape}")
    return S
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 2: Portfolio Performance 
#-------------------------------------------------------------------------------------
def portfolio_performance(weights, mu, S, risk_free_rate=RISK_FREE_RATE):
    """
    Returns annualised return, volatility, and Sharpe ratio for given weights.
    weights can be a dict or pd.Series mapping ticker → weight.
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights)
    weights = weights.reindex(mu.index).fillna(0)
    w = weights.values

    port_return = float(np.dot(w, mu.values))
    port_vol    = float(np.sqrt(w @ S.values @ w))
    sharpe      = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    return {
        "annual_return": round(port_return, 6),
        "annual_vol":    round(port_vol,    6),
        "sharpe_ratio":  round(sharpe,      6),
        "weights":       weights.sort_values(ascending=False).round(6),
    }
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 3: Core Optimisations
#-------------------------------------------------------------------------------------
def _build_ef(mu, S, weight_bounds=(0, 1)):
    return EfficientFrontier(mu, S, weight_bounds=weight_bounds)
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Each optimization target is wrapped in a function that builds the Efficient Frontier,
# runs the optimization, and computes performance metrics for the resulting portfolio.
#-------------------------------------------------------------------------------------
def optimize_max_sharpe(mu, S, risk_free_rate=RISK_FREE_RATE, weight_bounds=(0, 1)):
    """
    Tangency portfolio — maximises Sharpe Ratio.
    The point where a line from the risk-free rate is tangent to the
    efficient frontier.
    """
    ef = _build_ef(mu, S, weight_bounds)
    try:
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    except OptimizationError as e:
        log.warning(f"max_sharpe failed: {e}. Retrying...")
        ef = _build_ef(mu, S, weight_bounds)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    perf = portfolio_performance(ef.clean_weights(), mu, S, risk_free_rate)
    log.info(f"Max Sharpe → Return: {perf['annual_return']:.2%}, Vol: {perf['annual_vol']:.2%}, Sharpe: {perf['sharpe_ratio']:.3f}")
    return perf
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# The Global Minimum Variance (GMV) portfolio is the leftmost point on the efficient frontier,
# representing the lowest possible volatility. It relies only on the covariance structure
#-------------------------------------------------------------------------------------
def optimize_min_volatility(mu, S, risk_free_rate=RISK_FREE_RATE, weight_bounds=(0, 1)):
    """
    Global Minimum Variance (GMV) portfolio — the leftmost point on
    the efficient frontier. Relies only on covariance structure.
    """
    ef = _build_ef(mu, S, weight_bounds)
    ef.min_volatility()
    perf = portfolio_performance(ef.clean_weights(), mu, S, risk_free_rate)
    log.info(f"Min Vol → Return: {perf['annual_return']:.2%}, Vol: {perf['annual_vol']:.2%}, Sharpe: {perf['sharpe_ratio']:.3f}")
    return perf
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# The naive 1/N equal-weight portfolio is a common benchmark. It does not rely on any
# estimates and can outperform optimized portfolios out-of-sample due to estimation errors.
#-------------------------------------------------------------------------------------
def optimize_equal_weight(mu, S, risk_free_rate=RISK_FREE_RATE):
    """
    Naive 1/N equal-weight portfolio — used as a benchmark.
    """
    n       = len(mu)
    weights = pd.Series({t: 1.0 / n for t in mu.index})
    perf    = portfolio_performance(weights, mu, S, risk_free_rate)
    log.info(f"Equal Weight (1/N) → Return: {perf['annual_return']:.2%}, Vol: {perf['annual_vol']:.2%}, Sharpe: {perf['sharpe_ratio']:.3f}")
    return perf
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 4: Efficient Frontier
#-------------------------------------------------------------------------------------
def compute_efficient_frontier(mu, S, n_points=50, risk_free_rate=RISK_FREE_RATE, weight_bounds=(0, 1)):
    """
    Generate N portfolios tracing the efficient frontier by sweeping
    target returns from min-vol to max-return.

    Returns a DataFrame with columns:
        target_return | annual_vol | annual_return | sharpe_ratio | <ticker weights>
    """
    min_ret = float(mu.min()) * 1.05
    max_ret = float(mu.max()) * 0.95
    target_returns = np.linspace(min_ret, max_ret, n_points)

    rows = []
    for target in target_returns:
        try:
            ef = _build_ef(mu, S, weight_bounds)
            ef.efficient_return(target_return=target)
            cleaned = ef.clean_weights()
            perf    = portfolio_performance(cleaned, mu, S, risk_free_rate)
            row = {
                "target_return": round(target, 6),
                "annual_vol":    perf["annual_vol"],
                "annual_return": perf["annual_return"],
                "sharpe_ratio":  perf["sharpe_ratio"],
            }
            row.update({t: round(w, 6) for t, w in cleaned.items()})
            rows.append(row)
        except Exception as e:
            log.debug(f"Skipping frontier point at return={target:.4f}: {e}")

    df = pd.DataFrame(rows)
    log.info(f"Efficient frontier: {len(df)} valid points out of {n_points} targets")
    return df
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 5: Summary Table 
#-------------------------------------------------------------------------------------
def build_summary_table(max_sharpe, min_vol, equal_weight):
    rows = {
        "Max Sharpe":     max_sharpe,
        "Min Volatility": min_vol,
        "Equal Weight":   equal_weight,
    }
    records = []
    for name, perf in rows.items():
        records.append({
            "Strategy":        name,
            "Ann. Return":     f"{perf['annual_return']:.2%}",
            "Ann. Volatility": f"{perf['annual_vol']:.2%}",
            "Sharpe Ratio":    f"{perf['sharpe_ratio']:.3f}",
            "Top Holding":     perf["weights"].idxmax(),
            "Top Weight":      f"{perf['weights'].max():.1%}",
        })
    return pd.DataFrame(records).set_index("Strategy")
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 6: Main Entry Point 
#-------------------------------------------------------------------------------------
def run_markowitz(
    prices,
    risk_free_rate=RISK_FREE_RATE,
    n_frontier_points=50,
    cov_method="ledoit_wolf",
    ret_method="mean_historical_return",
    weight_bounds=(0, 1),
    verbose=True,
):
    """
    Full Markowitz pipeline in a single call.

    Parameters
    ----------
    prices            : pd.DataFrame of adjusted closing prices (from fetch_data.py)
    risk_free_rate    : float — annual risk-free rate (default: 6.5% RBI repo rate)
    n_frontier_points : int   — number of efficient frontier points (default 50)
    cov_method        : str   — 'ledoit_wolf' | 'sample_cov' | 'exp_cov'
    ret_method        : str   — 'mean_historical_return' | 'ema_historical_return'
    weight_bounds     : tuple — (min, max) weight per asset. (0,1) = long-only
    verbose           : bool  — print summary table to console

    Returns
    -------
    dict : {
        'mu'           : pd.Series      expected returns per asset
        'S'            : pd.DataFrame   covariance matrix
        'max_sharpe'   : dict           weights + performance metrics
        'min_vol'      : dict           weights + performance metrics
        'equal_weight' : dict           weights + performance metrics
        'frontier'     : pd.DataFrame   efficient frontier points
        'summary'      : pd.DataFrame   strategy comparison table
    }

    Example
    -------
    >>> from data.fetch_data import get_data
    >>> from models.markowitz import run_markowitz
    >>>
    >>> prices, _ = get_data(tickers=["RELIANCE.NS", "TCS.NS", "INFY.NS"])
    >>> results   = run_markowitz(prices)
    >>> print(results["summary"])
    >>> frontier  = results["frontier"]   # plug directly into Plotly chart
    """
    log.info("=" * 60)
    log.info("  MARKOWITZ OPTIMISATION — START")
    log.info("=" * 60)

    mu = compute_expected_returns(prices, method=ret_method)
    S  = compute_covariance(prices, method=cov_method)

    max_sharpe_port   = optimize_max_sharpe(mu, S, risk_free_rate, weight_bounds)
    min_vol_port      = optimize_min_volatility(mu, S, risk_free_rate, weight_bounds)
    equal_weight_port = optimize_equal_weight(mu, S, risk_free_rate)

    frontier = compute_efficient_frontier(
        mu, S,
        n_points=n_frontier_points,
        risk_free_rate=risk_free_rate,
        weight_bounds=weight_bounds,
    )

    summary = build_summary_table(max_sharpe_port, min_vol_port, equal_weight_port)

    if verbose:
        print("\n" + "=" * 60)
        print("  MARKOWITZ RESULTS")
        print("=" * 60)
        print(summary.to_string())
        print("\n  Max Sharpe Weights:")
        print(max_sharpe_port["weights"].to_string())
        print("\n  Min Vol Weights:")
        print(min_vol_port["weights"].to_string())
        print("=" * 60 + "\n")

    return {
        "mu":           mu,
        "S":            S,
        "max_sharpe":   max_sharpe_port,
        "min_vol":      min_vol_port,
        "equal_weight": equal_weight_port,
        "frontier":     frontier,
        "summary":      summary,
    }
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# CLI entry point 
#-------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fetch_data import get_data

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Run Markowitz portfolio optimisation")
    parser.add_argument("--tickers", nargs="+",
        default=[
    "AXISBANK.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "KOTAKBANK.NS",
    "PNB.NS",
    "SBIN.NS",   
    ])
    parser.add_argument("--start",            default="2020-01-01")
    parser.add_argument("--end",              default=None)
    parser.add_argument("--rfr",   type=float, default=RISK_FREE_RATE)
    parser.add_argument("--cov",              default="ledoit_wolf",
                        choices=["ledoit_wolf", "sample_cov", "exp_cov"])
    parser.add_argument("--frontier_points",  type=int, default=50)
    parser.add_argument("--refresh",          action="store_true")
    args = parser.parse_args()

    prices, _ = get_data(
        tickers=args.tickers,
        start=args.start,
        end=args.end or pd.Timestamp.today().strftime("%Y-%m-%d"),
        force_refresh=args.refresh,
    )

    run_markowitz(
        prices=prices,
        risk_free_rate=args.rfr,
        n_frontier_points=args.frontier_points,
        cov_method=args.cov,
        verbose=True,
    )
#-------------------------------------------------------------------------------------