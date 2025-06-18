import pandas as pd
import numpy as np
import ta

def calculate_macd_rsi_bb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates MACD, RSI and Bollinger Bands indicators.
    Returns the dataframe with added columns.
    """
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    return df

def generate_macd_rsi_bb_features(data_by_tframe: dict) -> dict:
    """
    Takes dictionary {timeframe: df} and returns a dict with extracted features.
    Output keys: macd_diff_30m, rsi_30m, bb_width_30m, bb_percent_30m, ...
    """
    feature_row = {}
    for tf, df in data_by_tframe.items():
        df = df.copy().reset_index(drop=True)
        df = calculate_macd_rsi_bb(df)
        last_row = df.iloc[-1]

        feature_row[f'macd_diff_{tf}'] = last_row['macd_diff']
        feature_row[f'rsi_{tf}'] = last_row['rsi']
        feature_row[f'bb_width_{tf}'] = last_row['bb_width']
        feature_row[f'bb_percent_{tf}'] = last_row['bb_percent']

    return feature_row

if __name__ == "__main__":
    from fetch_ohlcv_binance import fetch_ohlcv
    import ccxt

    symbol = "BTC/USDC"
    exchange = "binance"
    tf_map = {"30m": "30m", "1h": "1h", "4h": "4h"}
    since = ccxt.binance().parse8601("2024-01-01T00:00:00Z")

    data_by_tframe = {}
    for label, tf in tf_map.items():
        df = fetch_ohlcv(exchange, symbol, tf, since)
        data_by_tframe[label] = df

    features = generate_macd_rsi_bb_features(data_by_tframe)
    print(features)
