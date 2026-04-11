#-------------------------------------------------------------------------------------
"""
Downloads and caches historical OHLCV (Open, High, Low, Close, Volume) price data from Yahoo Finance
for a configurable list based on user inputs of equity tickers.

Usage:
    python data/fetch_data.py                    # uses DEFAULT_TICKERS
    python data/fetch_data.py --tickers RELIANCE.NS TCS.NS INFY.NS
    python data/fetch_data.py --start 2020-01-01 --end 2026-04-01
"""
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Importing required libraries
#-------------------------------------------------------------------------------------
import os
import argparse
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Logging 
#-------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Defaults
#-------------------------------------------------------------------------------------
DEFAULT_TICKERS = [
    "AXISBANK.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "KOTAKBANK.NS",
    "PNB.NS",
    "SBIN.NS",   
]

DEFAULT_START = "2020-01-01"
DEFAULT_END   = datetime.today().strftime("%Y-%m-%d")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Core Functions
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Fetch Prices 
#-------------------------------------------------------------------------------------
"""
    Downloads adjusted closing prices for the given tickers.
    Parameters
    ----------
    tickers     : list of Yahoo Finance ticker symbols
    start       : start date string 'YYYY-MM-DD'
    end         : end date string 'YYYY-MM-DD'
    auto_adjust : whether to use split/dividend-adjusted prices

    Returns
    -------
    pd.DataFrame : Date-indexed DataFrame of closing prices
    """
#-------------------------------------------------------------------------------------
def fetch_prices(
    tickers: list,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    log.info(f"Fetching {len(tickers)} tickers from {start} to {end}...")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )

    # Handle single vs multi-ticker response
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="all")

    # Forward-fill minor gaps (e.g. exchange holidays)
    missing = prices.isnull().sum()
    if missing.any():
        log.warning(f"Missing values detected:\n{missing[missing > 0]}")
        prices = prices.ffill().dropna()

    log.info(f"Downloaded {len(prices)} trading days × {len(prices.columns)} assets")
    log.info(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")

    return prices
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Compute Returns
#-------------------------------------------------------------------------------------
"""
    Computes daily percentage returns from price data.
    Parameters
    ----------
    prices : pd.DataFrame of closing prices

    Returns
    -------
    pd.DataFrame : daily returns (first row dropped)
"""
#-------------------------------------------------------------------------------------
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    
    returns = prices.pct_change().dropna()
    log.info(f"Computed daily returns: {returns.shape}")
    return returns
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Export data to Comma Separated Values (CSV) file
#-------------------------------------------------------------------------------------
"""
    Saves a DataFrame to the data/processed/ directory.
    Parameters
    ----------
    df       : DataFrame to save
    filename : output filename (e.g. 'prices_nse.csv')

    Returns
    -------
    str : full file path of saved CSV
"""
#-------------------------------------------------------------------------------------
def save_to_csv(df: pd.DataFrame, filename: str) -> str:
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    filepath = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(filepath)
    log.info(f"Saved → {filepath}")
    return filepath
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Load cached prices
#-------------------------------------------------------------------------------------
"""
    Loads cached prices from data/processed/.
    Parameters
    ----------
    filename : CSV filename in data/processed/

    Returns
    -------
    pd.DataFrame : price DataFrame with DatetimeIndex
"""
#-------------------------------------------------------------------------------------
def load_prices(filename: str) -> pd.DataFrame:
    
    filepath = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Cached data not found at {filepath}. "
            f"Run fetch_data.py first to download data."
        )
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    log.info(f"Loaded cached data: {df.shape} from {filepath}")
    return df
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Main function to get data (with caching)
#-------------------------------------------------------------------------------------
"""
    Main entry point — returns (prices, returns).
    Uses cached CSV if available unless force_refresh=True.
    Parameters
    ----------
    tickers        : ticker list
    start          : start date
    end            : end date
    cache_filename : name for the cached CSV file
    force_refresh  : if True, re-downloads even if cache exists

    Returns
    -------
    tuple : (prices DataFrame, returns DataFrame)
"""
#-------------------------------------------------------------------------------------
def get_data(
    tickers: list = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_filename: str = "prices.csv",
    force_refresh: bool = False,
) -> tuple:
    cache_path = os.path.join(PROCESSED_DIR, cache_filename)

    if os.path.exists(cache_path) and not force_refresh:
        log.info(f"Cache found — loading (use force_refresh=True to re-download)")
        prices = load_prices(cache_filename)
    else:
        prices = fetch_prices(tickers, start, end)
        save_to_csv(prices, cache_filename)

    returns = compute_returns(prices)
    return prices, returns
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Summary Stats
#-------------------------------------------------------------------------------------
"""Prints a clean performance summary of the fetched dataset."""
#-------------------------------------------------------------------------------------
def print_summary(prices: pd.DataFrame, returns: pd.DataFrame) -> None:
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * (252 ** 0.5)
    sharpe  = ann_ret / ann_vol
    total_r = (prices.iloc[-1] / prices.iloc[0] - 1)

    summary = pd.DataFrame({
        "Ann. Return":  ann_ret.map("{:.2%}".format),
        "Ann. Vol":     ann_vol.map("{:.2%}".format),
        "Sharpe":       sharpe.map("{:.3f}".format),
        "Total Return": total_r.map("{:.2%}".format),
    })

    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Trading Days : {len(prices)}")
    print(f"  Assets       : {len(prices.columns)}")
    print(f"  Date Range   : {prices.index[0].date()} → {prices.index[-1].date()}")
    print("=" * 60)
    print(summary.to_string())
    print("=" * 60 + "\n")
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# CLI entry point
#-------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch and cache equity price data via yfinance"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help="Yahoo Finance ticker symbols (e.g. RELIANCE.NS TCS.NS AAPL)"
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help="Start date YYYY-MM-DD (default: 2020-01-01)"
    )
    parser.add_argument(
        "--end", default=DEFAULT_END,
        help="End date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force re-download even if cache exists"
    )
    parser.add_argument(
        "--output", default="prices.csv",
        help="Output cache filename (default: prices.csv)"
    )

    args = parser.parse_args()

    prices, returns = get_data(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        cache_filename=args.output,
        force_refresh=args.refresh,
    )

    print_summary(prices, returns)
#-------------------------------------------------------------------------------------