import pandas as pd
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

def apply_macd_rsi_bb_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Υπολογίζει MACD, Bollinger Bands και RSI features σε OHLCV dataframe.
    Απαιτεί: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """

    df = df.copy()

    # MACD
    macd = MACD(close=df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd_line"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    # Breakout % πάνω ή κάτω από μπάντες
    df["bb_breakout_pct"] = 0.0
    df.loc[df["close"] > df["bb_upper"], "bb_breakout_pct"] = (
        df["close"] - df["bb_upper"]
    ) / df["bb_upper"]
    df.loc[df["close"] < df["bb_lower"], "bb_breakout_pct"] = (
        df["close"] - df["bb_lower"]
    ) / df["bb_lower"]

    # RSI
    rsi = RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi.rsi()

    df.dropna(inplace=True)
    return df
