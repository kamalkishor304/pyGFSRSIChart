import pandas as pd
from data import fetch_monthly_close


def calculate_ema(data: pd.Series, window: int = 21) -> pd.Series:
    return data.ewm(span=window, adjust=False).mean()


def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    delta = data.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def get_monthly_rsi_trend(symbol: str, window=14):
    try:
        monthly_close = fetch_monthly_close(symbol)
        if monthly_close.empty or monthly_close.shape[0] < 20:
            return None
        monthly_rsi = calculate_rsi(monthly_close, window)
        monthly_rsi_ema = calculate_ema(monthly_rsi)
        combined = pd.DataFrame({
            "Date": monthly_rsi.index,
            "RSI": monthly_rsi.values,
            "EMA": monthly_rsi_ema.values
        })
        combined = combined.dropna()
        if combined.empty or combined.shape[0] < 2:
            return None
        latest = combined.iloc[-1]
        prev = combined.iloc[-2]
        return {
            "SYMBOL": symbol,
            "Monthly RSI": float(latest["RSI"]),
            "Monthly RSI EMA": float(latest["EMA"]),
            "Prev RSI": float(prev["RSI"]),
            "Prev EMA": float(prev["EMA"]),
        }
    except Exception:
        return None
