import pandas as pd
import numpy as np


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def calculate_hma(df: pd.DataFrame, period: int) -> pd.Series:
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))

    wma_half = wma(df['close'], half_period)
    wma_full = wma(df['close'], period)
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    return hma


def calculate_slope(series: pd.Series, window: int = 3) -> pd.Series:
    return series.diff(window) / window


def generate_hull_features(tf_data: dict) -> pd.DataFrame:
    """
    tf_data: {
        '30m': DataFrame,
        '1h': DataFrame,
        '4h': DataFrame
    }
    Returns a single row DataFrame with latest HMA values and slopes.
    """
    features = {}

    for label, period in [('30m', 21), ('1h', 21), ('4h', 21)]:
        df = tf_data[label].copy()

        df[f'hma_{label}'] = calculate_hma(df, period)
        df[f'hma_{label}_slope'] = calculate_slope(df[f'hma_{label}'])

        latest_hma = df[f'hma_{label}'].iloc[-1]
        latest_slope = df[f'hma_{label}_slope'].iloc[-1]

        features[f'hma_{label}'] = latest_hma
        features[f'hma_{label}_slope'] = latest_slope

    return pd.DataFrame([features])


if __name__ == "__main__":
    # Example for testing
    df_30m = pd.read_csv("../../data/historical/binance/BTCUSDC/30m.csv")
    df_1h = pd.read_csv("../../data/historical/binance/BTCUSDC/1h.csv")
    df_4h = pd.read_csv("../../data/historical/binance/BTCUSDC/4h.csv")

    tf_data = {
        '30m': df_30m,
        '1h': df_1h,
        '4h': df_4h
    }

    output = generate_hull_features(tf_data)
    print(output)
