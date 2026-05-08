#-------------------------------------------------------------------------------------
"""
drawdown.py
-----------
Drawdown analysis and CDaR (Conditional Drawdown-at-Risk) optimisation.

Provides:-
  1. Drawdown curve       - underwater equity curve over time
  2. Max Drawdown (MDD)   - single worst peak-to-trough loss
  3. CDaR optimisation    - minimise average of worst β% drawdowns
  4. CDaR Efficient Frontier - sweep target returns under CDaR constraint
  5. Per-asset drawdown stats - for dashboard comparison table

Integrates directly with data/fetch_data.py and models/markowitz.py.

Usage (standalone):-
    from data.fetch_data import get_data
    from models.drawdown import run_drawdown

    prices, returns = get_data()
    results = run_drawdown(prices, returns)
    print(results["summary"])
"""
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Importing Required Libraries
#-------------------------------------------------------------------------------------
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
#-------------------------------------------------------------------------------------
log = logging.getLogger(__name__)
#-------------------------------------------------------------------------------------
TRADING_DAYS   = 252
RISK_FREE_RATE = 0.0525   # RBI repo rate, April 2026
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 1: Drawdown Curve
#-------------------------------------------------------------------------------------
"""
    Compute the drawdown (underwater) curve from a cumulative return series.
    At each point in time, drawdown = how far below the running peak you are.

    Parameters:- 
    cumulative_returns : pd.Series - cumulative portfolio value (starts at 1.0)
    
    Returns:- 
    pd.Series : drawdown values (0 = at peak, -0.20 = 20% below peak)

    Example:-
    If cumulative value goes [1.0, 1.1, 1.05, 1.2, 1.0]:
    Running peak           = [1.0, 1.1, 1.1,  1.2, 1.2]
    Drawdown               = [0.0, 0.0,-0.045, 0.0,-0.167]
"""
def compute_drawdown_curve(cumulative_returns: pd.Series) -> pd.Series:
    running_peak = cumulative_returns.cummax()
    drawdown     = (cumulative_returns - running_peak) / running_peak
    return drawdown
#-------------------------------------------------------------------------------------
"""
    Compute the drawdown curve for a weighted portfolio.

    Parameters:-
    returns : pd.DataFrame - daily returns per asset
    weights : dict or pd.Series - portfolio weights (must sum to 1)

    Returns: pd.Series : date-indexed drawdown curve
"""
def compute_portfolio_drawdown(
    returns: pd.DataFrame,
    weights: dict | pd.Series,
) -> pd.Series:
    if isinstance(weights, dict):
        weights = pd.Series(weights)

    weights = weights.reindex(returns.columns).fillna(0)
    port_returns = returns @ weights.values              # daily portfolio returns
    cum_returns  = (1 + port_returns).cumprod()          # cumulative growth curve
    drawdown     = compute_drawdown_curve(cum_returns)

    return drawdown
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 2: Drawdown Metrics 
#-------------------------------------------------------------------------------------
"""
    Maximum Drawdown (MDD) - the single worst peak-to-trough loss.
    Returns a positive float: 0.25 means a 25% max drawdown.
"""
def max_drawdown(drawdown_curve: pd.Series) -> float:   
    return float(abs(drawdown_curve.min()))
#-------------------------------------------------------------------------------------
"""
    Average Drawdown - mean of all drawdown observations.
    Gives a smoother picture of typical underwater periods vs MDD.
"""
def average_drawdown(drawdown_curve: pd.Series) -> float:
    return float(abs(drawdown_curve.mean()))
#-------------------------------------------------------------------------------------
"""
    Conditional Drawdown-at-Risk (CDaR) at confidence level beta.
    CDaR is the average of the worst (1-beta)% drawdown observations.
    It sits between Average Drawdown (beta=0) and Max Drawdown (beta=1).

    Parameters:-
    drawdown_curve : pd.Series - drawdown values (negative floats)
    beta           : float     - confidence level, typically 0.95

    Returns: float : CDaR value (positive, e.g. 0.18 = 18% average tail drawdown)

    Intuition:-
    beta=0.95 → "On the worst 5% of days, my average drawdown was X%"
    beta=0.00 → equivalent to Average Drawdown
    beta=1.00 → equivalent to Max Drawdown
"""
def cdar(drawdown_curve: pd.Series, beta: float = 0.95) -> float:
    sorted_dd = np.sort(drawdown_curve.values)           # ascending (most negative first)
    cutoff_idx = int(np.floor((1 - beta) * len(sorted_dd)))
    cutoff_idx = max(cutoff_idx, 1)                      # at least 1 observation
    tail_mean  = sorted_dd[:cutoff_idx].mean()
    return float(abs(tail_mean))
#-------------------------------------------------------------------------------------
"""
    Calmar Ratio = Annualised Return / Max Drawdown.
    Higher is better. Complements Sharpe by penalising crash severity
    rather than variance.

    Parameters:-
    returns        : pd.DataFrame daily returns
    weights        : portfolio weights
    risk_free_rate : not used in formula but kept for API consistency

    Returns: float : Calmar ratio (annualised return / max drawdown)
"""
def calmar_ratio(
    returns: pd.DataFrame,
    weights: dict | pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:    
    if isinstance(weights, dict):
        weights = pd.Series(weights)

    weights     = weights.reindex(returns.columns).fillna(0)
    port_returns = returns @ weights.values
    ann_return  = float(port_returns.mean() * TRADING_DAYS)
    cum_returns = (1 + port_returns).cumprod()
    dd_curve    = compute_drawdown_curve(cum_returns)
    mdd         = max_drawdown(dd_curve)

    return round(ann_return / mdd, 4) if mdd > 0 else np.inf
#-------------------------------------------------------------------------------------
"""
    Compute all drawdown metrics for a given portfolio in one call.
    Returns:-
    dict : {
        'drawdown_curve'   : pd.Series,
        'max_drawdown'     : float,
        'average_drawdown' : float,
        'cdar'             : float,
        'calmar_ratio'     : float,
        'beta'             : float,}
"""
def drawdown_stats(
    returns: pd.DataFrame,
    weights: dict | pd.Series,
    beta: float = 0.95,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:    
    dd_curve = compute_portfolio_drawdown(returns, weights)
    return {
        "drawdown_curve":   dd_curve,
        "max_drawdown":     round(max_drawdown(dd_curve), 6),
        "average_drawdown": round(average_drawdown(dd_curve), 6),
        "cdar":             round(cdar(dd_curve, beta), 6),
        "calmar_ratio":     calmar_ratio(returns, weights, risk_free_rate),
        "beta":             beta,}
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 3: CDaR Optimisation 
#-------------------------------------------------------------------------------------
"""
    Internal: compute CDaR for a weight array. Used by scipy optimiser.
"""
def _portfolio_cdar(weights_arr: np.ndarray, returns_arr: np.ndarray, beta: float) -> float: 
    port_returns = returns_arr @ weights_arr
    cum_returns  = np.cumprod(1 + port_returns)
    running_peak = np.maximum.accumulate(cum_returns)
    dd           = (cum_returns - running_peak) / running_peak
    sorted_dd    = np.sort(dd)
    cutoff_idx   = max(int(np.floor((1 - beta) * len(sorted_dd))), 1)
    return float(abs(sorted_dd[:cutoff_idx].mean()))
#-------------------------------------------------------------------------------------
"""
    Find portfolio weights that MINIMISE CDaR at confidence level beta.

    This is the CDaR equivalent of the Min-Volatility portfolio.
    It finds the allocation that minimises the average severity of
    the worst drawdown periods - regardless of expected return.

    Parameters:-
    returns      : pd.DataFrame - daily returns
    beta         : float        - CDaR confidence level (default 0.95)
    weight_bounds: tuple        - (min, max) per asset, default long-only
    risk_free_rate: float       - for Calmar ratio reporting

    Returns:-
    dict : {
        'weights'          : pd.Series,
        'cdar'             : float,
        'max_drawdown'     : float,
        'average_drawdown' : float,
        'calmar_ratio'     : float,
        'annual_return'    : float,
        'annual_vol'       : float,}
"""
def optimize_min_cdar(
    returns: pd.DataFrame,
    beta: float = 0.95,
    weight_bounds: tuple = (0.0, 1.0),
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    n           = returns.shape[1]
    returns_arr = returns.values
    tickers     = returns.columns.tolist()
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = [weight_bounds] * n
    w0          = np.array([1.0 / n] * n)
    result = minimize(
        fun     = lambda w: _portfolio_cdar(w, returns_arr, beta),
        x0      = w0,
        method  = "SLSQP",
        bounds  = bounds,
        constraints = constraints,
        options = {"maxiter": 1000, "ftol": 1e-9},
    )

    if not result.success:
        log.warning(f"CDaR minimisation did not fully converge: {result.message}")

    raw_weights = result.x
    raw_weights = np.clip(raw_weights, 0, 1)
    raw_weights /= raw_weights.sum()                    # re-normalise after clip

    weights_series = pd.Series(raw_weights, index=tickers).sort_values(ascending=False)

    # Performance metrics
    port_returns = returns @ raw_weights
    ann_return   = float(port_returns.mean() * TRADING_DAYS)
    ann_vol      = float(port_returns.std() * np.sqrt(TRADING_DAYS))
    dd_curve     = compute_portfolio_drawdown(returns, weights_series)

    stats = {
        "weights":          weights_series.round(6),
        "cdar":             round(cdar(dd_curve, beta), 6),
        "max_drawdown":     round(max_drawdown(dd_curve), 6),
        "average_drawdown": round(average_drawdown(dd_curve), 6),
        "calmar_ratio":     round(ann_return / max_drawdown(dd_curve), 4)
                            if max_drawdown(dd_curve) > 0 else np.inf,
        "annual_return":    round(ann_return, 6),
        "annual_vol":       round(ann_vol, 6),
        "sharpe_ratio":     round((ann_return - risk_free_rate) / ann_vol, 4)
                            if ann_vol > 0 else 0.0,}

    log.info(
        f"Min CDaR (β={beta}) → CDaR: {stats['cdar']:.2%}, "
        f"MDD: {stats['max_drawdown']:.2%}, Return: {stats['annual_return']:.2%}")
    
    return stats
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 4: CDaR Efficient Frontier 
#-------------------------------------------------------------------------------------
"""
    Generate N portfolios tracing the CDaR-efficient frontier.
    Sweeps target return levels and finds the minimum-CDaR portfolio
    achievable at each return - same logic as Markowitz frontier but
    using CDaR as the risk axis instead of volatility.

    Parameters:-
    returns      : pd.DataFrame - daily returns
    beta         : float        - CDaR confidence level (default 0.95)
    n_points     : int          - number of frontier points
    weight_bounds: tuple        - (min, max) weight per asset
    risk_free_rate: float       - for Sharpe ratio

    Returns:-
    pd.DataFrame : columns = [
        'target_return', 'cdar', 'max_drawdown',
        'annual_return', 'annual_vol', 'sharpe_ratio',
        + one weight column per asset]
"""
def compute_cdar_frontier(
    returns: pd.DataFrame,
    beta: float = 0.95,
    n_points: int = 40,
    weight_bounds: tuple = (0.0, 1.0),
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    n           = returns.shape[1]
    returns_arr = returns.values
    tickers     = returns.columns.tolist()

    # Determine return range using individual asset returns as bounds
    asset_ann_returns = returns.mean() * TRADING_DAYS
    min_ret = float(asset_ann_returns.min()) * 1.05
    max_ret = float(asset_ann_returns.max()) * 0.95
    target_returns = np.linspace(min_ret, max_ret, n_points)

    rows = []

    for target in target_returns:
        constraints = [
            {"type": "eq",  "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq","fun": lambda w, t=target: (returns_arr @ w).mean() * TRADING_DAYS - t},]
        bounds = [weight_bounds] * n
        w0     = np.array([1.0 / n] * n)

        try:
            result = minimize(
                fun         = lambda w: _portfolio_cdar(w, returns_arr, beta),
                x0          = w0,
                method      = "SLSQP",
                bounds      = bounds,
                constraints = constraints,
                options     = {"maxiter": 1000, "ftol": 1e-9},
            )

            if not result.success:
                log.debug(f"Skipping CDaR frontier point at return={target:.4f}: {result.message}")
                continue

            w = result.x
            w = np.clip(w, 0, 1)
            w /= w.sum()

            port_returns_arr = returns_arr @ w
            ann_ret = float(port_returns_arr.mean() * TRADING_DAYS)
            ann_vol = float(port_returns_arr.std() * np.sqrt(TRADING_DAYS))
            w_series = pd.Series(w, index=tickers)
            dd_curve = compute_portfolio_drawdown(returns, w_series)

            row = {
                "target_return": round(target, 6),
                "annual_return": round(ann_ret, 6),
                "annual_vol":    round(ann_vol, 6),
                "cdar":          round(cdar(dd_curve, beta), 6),
                "max_drawdown":  round(max_drawdown(dd_curve), 6),
                "sharpe_ratio":  round((ann_ret - risk_free_rate) / ann_vol, 4)
                                 if ann_vol > 0 else 0.0,
            }
            row.update({t: round(wt, 6) for t, wt in zip(tickers, w)})
            rows.append(row)

        except Exception as e:
            log.debug(f"CDaR frontier error at return={target:.4f}: {e}")
            continue

    df = pd.DataFrame(rows)
    log.info(f"CDaR frontier: {len(df)} valid points out of {n_points} targets")
    return df
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 5: Per-Asset Drawdown Stats
#-------------------------------------------------------------------------------------
"""
    Compute Max Drawdown and recovery stats for each individual asset.
    Useful for the dashboard's asset comparison panel.

    Parameters: prices : pd.DataFrame - adjusted closing prices

    Returns: pd.DataFrame : one row per asset with columns: ['Max Drawdown', 'Avg Drawdown', 'Current Drawdown', 'Ann. Return']
"""
def asset_drawdown_table(prices: pd.DataFrame) -> pd.DataFrame:
    records = []
    for ticker in prices.columns:
        cum = prices[ticker] / prices[ticker].iloc[0]
        dd  = compute_drawdown_curve(cum)
        ann_ret = float((prices[ticker].pct_change().dropna().mean()) * TRADING_DAYS)

        records.append({
            "Ticker":            ticker,
            "Max Drawdown":      f"{abs(dd.min()):.2%}",
            "Avg Drawdown":      f"{abs(dd.mean()):.2%}",
            "Current Drawdown":  f"{abs(dd.iloc[-1]):.2%}",
            "Ann. Return":       f"{ann_ret:.2%}",
        })
    return pd.DataFrame(records).set_index("Ticker")
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 6: Main Entry Point 
#-------------------------------------------------------------------------------------
"""
    Full drawdown analysis pipeline in a single call.

    Parameters:-
    prices             : pd.DataFrame - adjusted closing prices
    returns            : pd.DataFrame - daily returns (from fetch_data.py)
    markowitz_weights  : dict         - optional Max Sharpe weights from markowitz.py
                                        to compare against CDaR-optimal portfolio
    beta               : float        - CDaR confidence level (default 0.95)
    n_frontier_points  : int          - CDaR frontier points (default 40)
    weight_bounds      : tuple        - (min, max) per asset, default long-only
    risk_free_rate     : float        - RBI repo rate (default 5.25%)
    verbose            : bool         - print summary to console

    Returns:-
    dict : {
        'min_cdar'        : dict          - min-CDaR portfolio stats + weights
        'cdar_frontier'   : pd.DataFrame  - CDaR efficient frontier points
        'asset_table'     : pd.DataFrame  - per-asset drawdown stats
        'markowitz_dd'    : dict | None   - drawdown stats for Markowitz weights
                                            (only if markowitz_weights provided)
        'equal_weight_dd' : dict          - drawdown stats for 1/N benchmark}

    Example:-
    >>> from data.fetch_data import get_data
    >>> from models.markowitz import run_markowitz
    >>> from models.drawdown import run_drawdown
    >>>
    >>> prices, returns = get_data()
    >>> mk_results      = run_markowitz(prices)
    >>> dd_results      = run_drawdown(
    ...     prices,
    ...     returns,
    ...     markowitz_weights = mk_results["max_sharpe"]["weights"].to_dict()
    ... )
    >>> print(dd_results["min_cdar"]["cdar"])
    >>> frontier = dd_results["cdar_frontier"]   # plug into Plotly chart
"""
def run_drawdown(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    markowitz_weights: dict = None,
    beta: float = 0.95,
    n_frontier_points: int = 40,
    weight_bounds: tuple = (0.0, 1.0),
    risk_free_rate: float = RISK_FREE_RATE,
    verbose: bool = True,
) -> dict:
    log.info("=" * 60)
    log.info("  DRAWDOWN ANALYSIS - START")
    log.info("=" * 60)

    # 1. Min-CDaR portfolio 
    min_cdar_port = optimize_min_cdar(returns, beta, weight_bounds, risk_free_rate)

    # 2. CDaR frontier
    frontier = compute_cdar_frontier(
        returns, beta, n_frontier_points, weight_bounds, risk_free_rate)

    # 3. Per-asset table
    asset_table = asset_drawdown_table(prices)

    # 4. Equal-weight benchmark
    n        = returns.shape[1]
    eq_w     = {t: 1.0 / n for t in returns.columns}
    eq_dd    = drawdown_stats(returns, eq_w, beta, risk_free_rate)

    # 5. Optional: Markowitz comparison
    mk_dd = None
    if markowitz_weights is not None:
        mk_dd = drawdown_stats(returns, markowitz_weights, beta, risk_free_rate)
        log.info(
            f"Markowitz (Max Sharpe) drawdown → "
            f"MDD: {mk_dd['max_drawdown']:.2%}, CDaR: {mk_dd['cdar']:.2%}"
        )

    if verbose:
        print("\n" + "=" * 60)
        print("  DRAWDOWN RESULTS")
        print("=" * 60)

        print(f"\n  Min-CDaR Portfolio (β={beta}):")
        print(f"    CDaR:           {min_cdar_port['cdar']:.2%}")
        print(f"    Max Drawdown:   {min_cdar_port['max_drawdown']:.2%}")
        print(f"    Calmar Ratio:   {min_cdar_port['calmar_ratio']:.3f}")
        print(f"    Annual Return:  {min_cdar_port['annual_return']:.2%}")
        print(f"    Annual Vol:     {min_cdar_port['annual_vol']:.2%}")
        print(f"\n  Weights:")
        print(min_cdar_port["weights"].to_string())

        print(f"\n  Equal Weight (1/N) Benchmark:")
        print(f"    CDaR:           {eq_dd['cdar']:.2%}")
        print(f"    Max Drawdown:   {eq_dd['max_drawdown']:.2%}")

        if mk_dd:
            print(f"\n  Markowitz Max Sharpe Drawdown Profile:")
            print(f"    CDaR:           {mk_dd['cdar']:.2%}")
            print(f"    Max Drawdown:   {mk_dd['max_drawdown']:.2%}")
            print(f"    Calmar Ratio:   {mk_dd['calmar_ratio']:.3f}")

        print(f"\n  Per-Asset Drawdown Table:")
        print(asset_table.to_string())
        print("=" * 60 + "\n")

    return {
        "min_cdar":        min_cdar_port,
        "cdar_frontier":   frontier,
        "asset_table":     asset_table,
        "markowitz_dd":    mk_dd,
        "equal_weight_dd": eq_dd,}
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# CLI 
#-------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data.fetch_data import get_data
    from models.markowitz import run_markowitz

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Run CDaR drawdown analysis")
    parser.add_argument("--tickers", nargs="+",
        default=["AXISBANK.NS", "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "PNB.NS", "SBIN.NS",])
    parser.add_argument("--start",   default="2020-01-01")
    parser.add_argument("--end",     default=None)
    parser.add_argument("--beta",    type=float, default=0.95)
    parser.add_argument("--points",  type=int,   default=40)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    prices, returns = get_data(
        tickers=args.tickers,
        start=args.start,
        end=args.end or pd.Timestamp.today().strftime("%Y-%m-%d"),
        force_refresh=args.refresh,)

    mk = run_markowitz(prices, verbose=False)

    run_drawdown(
        prices=prices,
        returns=returns,
        markowitz_weights=mk["max_sharpe"]["weights"].to_dict(),
        beta=args.beta,
        n_frontier_points=args.points,
        verbose=True,
    )
#-------------------------------------------------------------------------------------