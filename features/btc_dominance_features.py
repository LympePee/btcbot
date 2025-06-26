
from btcbot.utils.fetch_binance import fetch_ohlcv_binance
from datetime import datetime, timedelta
import pandas as pd

LOOKBACK_HOURS = 48
DOMINANCE_SYMBOL = "BTC/USDT"

def get_btc_dominance_features(since_timestamp=None):
    if since_timestamp is None:
        now = datetime.utcnow()
        since = int((now - timedelta(hours=LOOKBACK_HOURS)).timestamp() * 1000)
    else:
        since = since_timestamp

    df = fetch_ohlcv_binance(DOMINANCE_SYMBOL, "1h", since)

    if df.empty:
        return {}

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)

    df["d_change"] = df["close"].pct_change().fillna(0) * 100
    avg_change = df["d_change"].mean()
    std_dev = df["d_change"].std()

    return {
        "btc_d_avg_change": avg_change,
        "btc_d_volatility": std_dev,
        "btc_d_change": df["d_change"].iloc[-1]
    }
