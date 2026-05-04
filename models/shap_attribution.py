#-------------------------------------------------------------------------------------
"""
shap_attribution.py
-------------------
SHAP (SHapley Additive exPlanations) attribution for portfolio weight decisions.
Trains a gradient boosting model that learns to predict portfolio weights from asset-level 
financial features, then uses SHAP to explain WHY the optimiser allocated capital the way it did.

Pipeline:-
  1. Feature engineering  — compute financial features per asset per period
  2. Target creation      — use CDaR-optimal weights as labels
  3. Model training       — XGBoost regressor per asset
  4. SHAP computation     — global + local explanations
  5. Attribution output   — ready to plug into dashboard charts

Integrates with:-
  - data/fetch_data.py    → prices, returns
  - models/markowitz.py   → mu, S, markowitz weights
  - models/drawdown.py    → cdar weights, drawdown curve

Usage (standalone):-
    from data.fetch_data import get_data
    from models.markowitz import run_markowitz
    from models.drawdown import run_drawdown
    from models.shap_attribution import run_shap

    prices, returns = get_data()
    mk = run_markowitz(prices, verbose=False)
    dd = run_drawdown(prices, returns, verbose=False)
    results = run_shap(prices, returns, mk, dd)
"""
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Importing Required Libraries
#-------------------------------------------------------------------------------------
import logging
import warnings
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
#-------------------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger(__name__)
#-------------------------------------------------------------------------------------
TRADING_DAYS   = 252
RISK_FREE_RATE = 0.0525   # RBI repo rate April 2026
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 1: Feature Engineering 
#-------------------------------------------------------------------------------------
def compute_asset_features(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    window_short: int = 21,    # ~1 month
    window_long:  int = 63,    # ~3 months
) -> pd.DataFrame:
    """
    Compute rolling financial features for each asset at each point in time.

    These features capture what the optimiser "sees" about each stock:
    momentum, volatility, drawdown severity, return consistency, and
    risk-adjusted performance. The SHAP model learns which of these
    drove the final weight allocation.

    Parameters:-
    prices       : pd.DataFrame — adjusted closing prices
    returns      : pd.DataFrame — daily returns
    window_short : int          — short rolling window in trading days (~1 month)
    window_long  : int          — long rolling window in trading days (~3 months)

    Returns: pd.DataFrame : MultiIndex columns (feature, ticker) — shape (T, F*N)

    Features computed per asset:-
    momentum_short    : cumulative return over short window
    momentum_long     : cumulative return over long window
    volatility_short  : annualised rolling volatility (short)
    volatility_long   : annualised rolling volatility (long)
    sharpe_rolling    : rolling Sharpe ratio (short window, excess over RFR)
    max_drawdown_roll : rolling max drawdown (long window)
    avg_drawdown_roll : rolling average drawdown (long window)
    skewness          : rolling return skewness (long window)
    downside_vol      : annualised downside deviation (returns below 0)
    sortino_rolling   : rolling Sortino ratio (short window)
    """
    features = {}

    for ticker in returns.columns:
        r = returns[ticker]
        p = prices[ticker]

        # Momentum
        mom_short = (1 + r).rolling(window_short).apply(np.prod, raw=True) - 1
        mom_long  = (1 + r).rolling(window_long).apply(np.prod, raw=True) - 1

        # Volatility
        vol_short = r.rolling(window_short).std() * np.sqrt(TRADING_DAYS)
        vol_long  = r.rolling(window_long).std()  * np.sqrt(TRADING_DAYS)

        # Rolling Sharpe (annualised, excess over daily RFR)
        daily_rfr   = RISK_FREE_RATE / TRADING_DAYS
        excess_r    = r - daily_rfr
        sharpe_roll = (
            excess_r.rolling(window_short).mean() * TRADING_DAYS
        ) / (r.rolling(window_short).std() * np.sqrt(TRADING_DAYS))

        # Rolling Max Drawdown
        cum_long   = (1 + r).rolling(window_long).apply(
            lambda x: (np.cumprod(x) / np.maximum.accumulate(np.cumprod(x)) - 1).min(),
            raw=True)
        mdd_roll   = cum_long.abs()

        # Rolling Average Drawdown
        avg_dd_roll = (1 + r).rolling(window_long).apply(
            lambda x: abs((np.cumprod(x) / np.maximum.accumulate(np.cumprod(x)) - 1).mean()),
            raw=True)

        # Skewness
        skew_roll = r.rolling(window_long).skew()

        # Downside Volatility (Sortino denominator)
        downside   = r.copy()
        downside[downside > 0] = 0
        down_vol   = downside.rolling(window_short).std() * np.sqrt(TRADING_DAYS)

        # Sortino Ratio
        sortino    = (
            excess_r.rolling(window_short).mean() * TRADING_DAYS
        ) / down_vol.replace(0, np.nan)

        features[ticker] = pd.DataFrame({
            "momentum_short":    mom_short,
            "momentum_long":     mom_long,
            "volatility_short":  vol_short,
            "volatility_long":   vol_long,
            "sharpe_rolling":    sharpe_roll,
            "max_drawdown_roll": mdd_roll,
            "avg_drawdown_roll": avg_dd_roll,
            "skewness":          skew_roll,
            "downside_vol":      down_vol,
            "sortino_rolling":   sortino,})

    # Stack into MultiIndex DataFrame: (date, ticker) rows
    feature_df = pd.concat(features, axis=1)   # columns: (ticker, feature)
    feature_df.columns = feature_df.columns.swaplevel(0, 1)
    feature_df = feature_df.sort_index(axis=1)

    log.info(f"Features computed: {feature_df.shape[1]} columns "
             f"({len(returns.columns)} assets × 10 features)")
    return feature_df
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 2: Build Training Dataset 
#-------------------------------------------------------------------------------------
"""
    Build X (features) and y (target weights) for model training.
    The model learns: given the financial features of each asset at time t,
    what weight did the CDaR optimiser assign to it?

    We create one row per asset per rolling window period, treating each
    asset as a separate observation. This gives the model enough samples
    to learn feature → weight relationships.

    Parameters:-
    feature_df        : pd.DataFrame   — from compute_asset_features()
    cdar_weights      : dict           — CDaR-optimal weights {ticker: weight}
    markowitz_weights : dict           — Markowitz Max Sharpe weights {ticker: weight}
    returns           : pd.DataFrame   — daily returns
    lookback          : int            — rolling window for target weight assignment

    Returns:-
    tuple : (X, y_cdar, y_markowitz, feature_names, tickers, dates)
"""
def build_training_data(
    feature_df: pd.DataFrame,
    cdar_weights: dict,
    markowitz_weights: dict,
    returns: pd.DataFrame,
    lookback: int = 63,
) -> tuple:
    
    tickers       = returns.columns.tolist()
    feature_names = feature_df.columns.get_level_values(0).unique().tolist()

    rows_X          = []
    rows_y_cdar     = []
    rows_y_markowitz = []
    row_meta        = []    # (date, ticker)

    # Use rolling windows - each window end date is one training sample
    dates = feature_df.dropna().index

    for date in dates:
        for ticker in tickers:
            try:
                row = feature_df.loc[date, (slice(None), ticker)]
                row.index = row.index.get_level_values(0)

                if row.isnull().any():
                    continue

                rows_X.append(row.values)
                rows_y_cdar.append(cdar_weights.get(ticker, 0.0))
                rows_y_markowitz.append(markowitz_weights.get(ticker, 0.0))
                row_meta.append((date, ticker))

            except (KeyError, Exception):
                continue

    X            = pd.DataFrame(rows_X, columns=feature_names)
    y_cdar       = pd.Series(rows_y_cdar,      name="cdar_weight")
    y_markowitz  = pd.Series(rows_y_markowitz, name="markowitz_weight")
    meta_df      = pd.DataFrame(row_meta, columns=["date", "ticker"])

    # Add ticker as a categorical feature (one-hot encoded)
    ticker_dummies = pd.get_dummies(meta_df["ticker"], prefix="ticker")
    X = pd.concat([X.reset_index(drop=True), ticker_dummies.reset_index(drop=True)], axis=1)

    log.info(f"Training dataset: {X.shape[0]} rows × {X.shape[1]} features")
    return X, y_cdar, y_markowitz, feature_names, tickers, meta_df
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 3: Train XGBoost Model
#-------------------------------------------------------------------------------------
"""
    Train an XGBoost regressor to predict portfolio weights from features.
    XGBoost is used because:
      - SHAP has native, exact support for tree-based models (TreeExplainer)
      - It handles feature interactions naturally
      - No feature scaling required (though we scale for reporting)
      - Fast to train on tabular financial data

    Parameters:-
    X            : pd.DataFrame — feature matrix
    y            : pd.Series    — target weights
    label        : str          — 'CDaR' or 'Markowitz' for logging
    test_size    : float        — fraction held out for evaluation
    random_state : int          — reproducibility seed

    Returns: tuple : (model, X_train, X_test, y_train, y_test, metrics_dict)
"""
def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    label: str = "CDaR",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple: 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True)

    model = XGBRegressor(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 5,
        reg_alpha         = 0.1,     # L1 regularisation
        reg_lambda        = 1.0,     # L2 regularisation
        random_state      = random_state,
        n_jobs            = -1,
        verbosity         = 0,)

    model.fit(
        X_train, y_train,
        eval_set    = [(X_test, y_test)],
        verbose     = False,)

    y_pred   = model.predict(X_test)
    r2       = r2_score(y_test, y_pred)
    mae      = mean_absolute_error(y_test, y_pred)

    metrics = {
        "r2_score": round(r2,  4),
        "mae":      round(mae, 6),
        "n_train":  len(X_train),
        "n_test":   len(X_test),}

    log.info(
        f"[{label}] XGBoost trained → R²: {r2:.4f}, MAE: {mae:.6f} "
        f"(train: {len(X_train)}, test: {len(X_test)})")
    return model, X_train, X_test, y_train, y_test, metrics
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 4: SHAP Computation
#-------------------------------------------------------------------------------------
"""
    Compute SHAP values using TreeExplainer (exact, fast for XGBoost).
    TreeExplainer is preferred over KernelExplainer because:
      - It computes EXACT Shapley values (not approximations)
      - 1000x faster for tree-based models
      - Handles feature interactions via TreeSHAP algorithm

    Parameters:-
    model : trained XGBRegressor
    X     : pd.DataFrame — feature matrix (same columns as training)
    label : str          — for logging

    Returns:-
    tuple : (explainer, shap_values_array, shap_df)
      - shap_values_array : np.ndarray shape (n_samples, n_features)
      - shap_df           : pd.DataFrame with SHAP values, same index as X
"""
def compute_shap_values(
    model: XGBRegressor,
    X: pd.DataFrame,
    label: str = "CDaR",
) -> tuple:   
    explainer    = shap.TreeExplainer(model)
    shap_values  = explainer.shap_values(X)

    shap_df = pd.DataFrame(
        shap_values,
        columns = X.columns,
        index   = X.index,
    )

    log.info(f"[{label}] SHAP values computed: {shap_df.shape}")
    return explainer, shap_values, shap_df
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 5: Attribution Summaries
#-------------------------------------------------------------------------------------
"""
    Global SHAP importance — mean absolute SHAP value per feature.
    This answers: "Across all assets and all time periods, which features
    most consistently drove the weight allocation decisions?"

    Parameters:-
    shap_df : pd.DataFrame — SHAP values from compute_shap_values()
    top_n   : int          — return top N features

    Returns: pd.DataFrame : columns ['feature', 'mean_abs_shap'] sorted descending
"""
def global_feature_importance(
    shap_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    df = mean_abs.reset_index()
    df.columns = ["feature", "mean_abs_shap"]
    df["mean_abs_shap"] = df["mean_abs_shap"].round(6)
    return df.head(top_n)
#-------------------------------------------------------------------------------------
"""
    Per-ticker mean SHAP values for core financial features only (excludes one-hot ticker dummies).
    This answers: "For each stock, which features most influenced
    its weight allocation?"

    Parameters:-
    shap_df       : pd.DataFrame — SHAP values
    meta_df       : pd.DataFrame — (date, ticker) metadata
    feature_names : list         — core feature names (not one-hot columns)

    Returns: pd.DataFrame : rows = tickers, columns = features, values = mean abs SHAP
"""
def per_ticker_shap(
    shap_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    feature_names: list,
) -> pd.DataFrame:
    core_cols   = [c for c in shap_df.columns if c in feature_names]
    shap_core   = shap_df[core_cols].copy()
    shap_core["ticker"] = meta_df["ticker"].values

    ticker_shap = (
        shap_core.groupby("ticker")[core_cols]
        .apply(lambda x: x.abs().mean())
        .round(6))

    return ticker_shap
#-------------------------------------------------------------------------------------
"""
    Signed mean SHAP values per ticker — captures directionality.
    A positive value means the feature INCREASED the allocation to that asset.
    A negative value means the feature DECREASED the allocation.

    Returns: pd.DataFrame : rows = tickers, columns = features, values = signed mean SHAP
"""
def shap_direction_table(
    shap_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    feature_names: list,
) -> pd.DataFrame:  
    core_cols          = [c for c in shap_df.columns if c in feature_names]
    shap_core          = shap_df[core_cols].copy()
    shap_core["ticker"] = meta_df["ticker"].values

    direction_table = (
        shap_core.groupby("ticker")[core_cols]
        .mean()
        .round(6))
    return direction_table
#-------------------------------------------------------------------------------------
"""
    Extract SHAP waterfall data for a single ticker's most recent observation.
    Used by the dashboard to render a waterfall chart explaining the
    latest weight decision for a chosen stock.

    Parameters:-
    explainer     : shap.TreeExplainer
    X             : pd.DataFrame
    meta_df       : pd.DataFrame — (date, ticker) metadata
    ticker        : str          — which ticker to explain
    feature_names : list         — core feature names

    Returns:-
    dict : {
        'ticker'     : str,
        'date'       : pd.Timestamp,
        'base_value' : float  — model's average prediction
        'features'   : list of feature names
        'shap_values': list of SHAP values
        'feature_values': list of actual feature values
    }
"""
def waterfall_data(
    explainer,
    X: pd.DataFrame,
    meta_df: pd.DataFrame,
    ticker: str,
    feature_names: list,
) -> dict:    
    ticker_mask = meta_df["ticker"] == ticker
    if not ticker_mask.any():
        log.warning(f"Ticker {ticker} not found in meta_df")
        return {}

    # Get the most recent row for this ticker
    ticker_idx  = meta_df[ticker_mask].index[-1]
    X_row       = X.loc[[ticker_idx]]
    date        = meta_df.loc[ticker_idx, "date"]

    shap_vals   = explainer.shap_values(X_row)[0]
    feature_vals = X_row.values[0]
    base_val    = explainer.expected_value

    # Filter to core features only for clean display
    core_mask   = [c in feature_names for c in X.columns]
    core_cols   = [c for c, m in zip(X.columns, core_mask) if m]
    core_shap   = [v for v, m in zip(shap_vals, core_mask) if m]
    core_fvals  = [v for v, m in zip(feature_vals, core_mask) if m]

    # Sort by absolute SHAP value
    sorted_idx  = np.argsort(np.abs(core_shap))[::-1]

    return {
        "ticker":         ticker,
        "date":           date,
        "base_value":     float(base_val),
        "features":       [core_cols[i] for i in sorted_idx],
        "shap_values":    [core_shap[i] for i in sorted_idx],
        "feature_values": [core_fvals[i] for i in sorted_idx],}
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 6: Print Summary
#-------------------------------------------------------------------------------------
"""Print a formatted SHAP attribution summary to console."""
def print_shap_summary(
    global_importance: pd.DataFrame,
    ticker_shap: pd.DataFrame,
    direction_table: pd.DataFrame,
    cdar_metrics: dict,
    markowitz_metrics: dict,
) -> None:
    print("\n" + "=" * 60)
    print("  SHAP ATTRIBUTION SUMMARY")
    print("=" * 60)

    print("\n  Model Performance:")
    print(f"    CDaR model       → R²: {cdar_metrics['r2_score']:.4f}, "
          f"MAE: {cdar_metrics['mae']:.6f}")
    print(f"    Markowitz model  → R²: {markowitz_metrics['r2_score']:.4f}, "
          f"MAE: {markowitz_metrics['mae']:.6f}")

    print("\n  Top Global Feature Importances (CDaR model):")
    for _, row in global_importance.iterrows():
        bar = "█" * int(row["mean_abs_shap"] * 500)
        print(f"    {row['feature']:22s}  {row['mean_abs_shap']:.6f}  {bar}")

    print("\n  Signed SHAP Direction per Ticker:")
    print("  (+ = feature increased allocation, - = decreased)")
    print(direction_table.to_string())
    print("=" * 60 + "\n")
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Step 7: Main Entry Point 
#-------------------------------------------------------------------------------------

"""
    Full SHAP attribution pipeline in a single call.

    Parameters:-
    prices             : pd.DataFrame — adjusted closing prices
    returns            : pd.DataFrame — daily returns
    markowitz_results  : dict         — output of run_markowitz()
    drawdown_results   : dict         — output of run_drawdown()
    top_n_features     : int          — top N features to report (default 10)
    window_short       : int          — short rolling window (default 21 days)
    window_long        : int          — long rolling window (default 63 days)
    verbose            : bool         — print summary to console

    Returns:-
    dict : {
        'feature_df'         : pd.DataFrame  — raw computed features
        'X'                  : pd.DataFrame  — training feature matrix
        'y_cdar'             : pd.Series     — CDaR weight targets
        'y_markowitz'        : pd.Series     — Markowitz weight targets
        'meta_df'            : pd.DataFrame  — (date, ticker) metadata
        'cdar_model'         : XGBRegressor  — trained CDaR model
        'markowitz_model'    : XGBRegressor  — trained Markowitz model
        'cdar_metrics'       : dict          — R², MAE for CDaR model
        'markowitz_metrics'  : dict          — R², MAE for Markowitz model
        'cdar_explainer'     : shap.TreeExplainer
        'cdar_shap_values'   : np.ndarray
        'cdar_shap_df'       : pd.DataFrame
        'global_importance'  : pd.DataFrame  — top features by mean |SHAP|
        'ticker_shap'        : pd.DataFrame  — per-ticker mean |SHAP|
        'direction_table'    : pd.DataFrame  — signed SHAP per ticker
        'feature_names'      : list
    }

    Example:-
    >>> from data.fetch_data import get_data
    >>> from models.markowitz import run_markowitz
    >>> from models.drawdown import run_drawdown
    >>> from models.shap_attribution import run_shap
    >>>
    >>> prices, returns = get_data()
    >>> mk = run_markowitz(prices, verbose=False)
    >>> dd = run_drawdown(prices, returns, verbose=False)
    >>> shap_results = run_shap(prices, returns, mk, dd)
    >>>
    >>> # Global importance → plug into bar chart
    >>> print(shap_results["global_importance"])
    >>>
    >>> # Waterfall for a specific ticker
    >>> from models.shap_attribution import waterfall_data
    >>> wf = waterfall_data(
    ...     shap_results["cdar_explainer"],
    ...     shap_results["X"],
    ...     shap_results["meta_df"],
    ...     ticker="SBIN.NS",
    ...     feature_names=shap_results["feature_names"])
"""
def run_shap(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    markowitz_results: dict,
    drawdown_results: dict,
    top_n_features: int = 10,
    window_short: int = 21,
    window_long: int = 63,
    verbose: bool = True,
) -> dict:    
    log.info("=" * 60)
    log.info("  SHAP ATTRIBUTION — START")
    log.info("=" * 60)

    # Extract weights from upstream results
    cdar_weights      = drawdown_results["min_cdar"]["weights"].to_dict()
    markowitz_weights = markowitz_results["max_sharpe"]["weights"].to_dict()

    # 1. Feature engineering
    feature_df = compute_asset_features(prices, returns, window_short, window_long)

    # 2. Build training dataset
    X, y_cdar, y_markowitz, feature_names, tickers, meta_df = build_training_data(
        feature_df, cdar_weights, markowitz_weights, returns)

    # 3. Train models
    cdar_model, X_train_c, X_test_c, y_train_c, y_test_c, cdar_metrics = train_model(
        X, y_cdar, label="CDaR")
    mk_model, X_train_m, X_test_m, y_train_m, y_test_m, mk_metrics = train_model(
        X, y_markowitz, label="Markowitz")

    # 4. SHAP values
    cdar_explainer, cdar_shap_vals, cdar_shap_df = compute_shap_values(
        cdar_model, X, label="CDaR")

    # 5. Attribution summaries
    global_imp    = global_feature_importance(cdar_shap_df, top_n=top_n_features)
    ticker_shap   = per_ticker_shap(cdar_shap_df, meta_df, feature_names)
    direction_tbl = shap_direction_table(cdar_shap_df, meta_df, feature_names)

    if verbose:
        print_shap_summary(
            global_imp, ticker_shap, direction_tbl,
            cdar_metrics, mk_metrics)

    log.info("SHAP attribution complete.")

    return {
        "feature_df":        feature_df,
        "X":                 X,
        "y_cdar":            y_cdar,
        "y_markowitz":       y_markowitz,
        "meta_df":           meta_df,
        "cdar_model":        cdar_model,
        "markowitz_model":   mk_model,
        "cdar_metrics":      cdar_metrics,
        "markowitz_metrics": mk_metrics,
        "cdar_explainer":    cdar_explainer,
        "cdar_shap_values":  cdar_shap_vals,
        "cdar_shap_df":      cdar_shap_df,
        "global_importance": global_imp,
        "ticker_shap":       ticker_shap,
        "direction_table":   direction_tbl,
        "feature_names":     feature_names,
    }
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# CLI
#-------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data.fetch_data import get_data
    from models.markowitz import run_markowitz
    from models.drawdown import run_drawdown

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="Run SHAP attribution analysis")
    parser.add_argument("--tickers", nargs="+",
        default=["AXISBANK.NS", "HDFCBANK.NS", "ICICIBANK.NS",
                 "KOTAKBANK.NS", "PNB.NS", "SBIN.NS"])
    parser.add_argument("--start",   default="2020-01-01")
    parser.add_argument("--end",     default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--top_n",   type=int, default=10)
    args = parser.parse_args()

    prices, returns = get_data(
        tickers=args.tickers,
        start=args.start,
        end=args.end or pd.Timestamp.today().strftime("%Y-%m-%d"),
        force_refresh=args.refresh,)

    mk = run_markowitz(prices, verbose=False)
    dd = run_drawdown(prices, returns,
                      markowitz_weights=mk["max_sharpe"]["weights"].to_dict(),
                      verbose=False)

    run_shap(prices, returns, mk, dd, top_n_features=args.top_n, verbose=True)
#-------------------------------------------------------------------------------------