import yfinance as yf
import pandas as pd
from typing import cast
from concurrent.futures import ThreadPoolExecutor, as_completed


def _fetch_quote(symbol: str):
    ticker = yf.Ticker(f"{symbol}.NS")
    try:
        hist = ticker.history(period="3d", interval="1d", auto_adjust=False)
        if hist.empty:
            return symbol, None, None
        # Use the last close as LTP and difference from previous close as change
        last = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2] if len(hist) > 1 else last
        change = last - prev
        return symbol, float(last), float(change)
    except Exception:
        return symbol, None, None


def fetch_quotes(symbols, max_workers=8):
    """Fetch LTP and absolute change for a list of symbols using yfinance.
    Returns a dict mapping symbol -> (ltp, change)
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_quote, s): s for s in symbols}
        for fut in as_completed(futures):
            sym, ltp, change = fut.result()
            if ltp is not None:
                results[sym] = (ltp, change)
    return results


def get_ltp_and_change(symbol):
    sym, ltp, change = _fetch_quote(symbol)
    return ltp, change


def fetch_monthly_close(symbol, period="10y") -> pd.Series:
    """Return monthly close series for the given symbol."""
    df = yf.download(f"{symbol}.NS", period=period, interval="1mo", progress=False)
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    # Flatten MultiIndex columns (yfinance returns MultiIndex for single-ticker download)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    monthly_close = df['Close']
    # If df['Close'] is a DataFrame (multi-column), take the first column.
    # Use iloc to avoid passing axis arguments that static checkers may flag.
    if isinstance(monthly_close, pd.DataFrame):
        monthly_close = monthly_close.iloc[:, 0]  # type: ignore

    # Tell type checkers this is a Series, then return cleaned floats
    monthly_close = cast(pd.Series, monthly_close)
    return monthly_close.dropna().astype(float)
