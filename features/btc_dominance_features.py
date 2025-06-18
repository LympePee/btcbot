# features/btc_dominance_features.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from scipy.stats import linregress


def fetch_btc_dominance_history(days: int = 30) -> pd.DataFrame:
    url = f"https://api.coingecko.com/api/v3/global"
    data = []
    
    for i in range(days):
        date = (datetime.utcnow() - timedelta(days=i)).strftime("%d-%m-%Y")
        resp = requests.get(url)
        if resp.status_code == 200:
            try:
                btc_dom = resp.json()["data"]["market_cap_percentage"]["btc"]
                timestamp = datetime.utcnow() - timedelta(days=i)
                data.append([timestamp, btc_dom])
            except:
                continue
        else:
            break

    df = pd.DataFrame(data, columns=["datetime", "btc_dominance"])
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def calculate_slope(series: pd.Series) -> float:
    x = np.arange(len(series))
    y = series.values
    slope, _, _, _, _ = linregress(x, y)
    return slope


def resample_and_generate_features(df: pd.DataFrame) -> dict:
    features = {}

    for tf, rule in {
        "30m": "30T",
        "1h": "1H",
        "4h": "4H"
    }.items():
        resampled = df.set_index("datetime").resample(rule).mean().dropna()
        resampled["slope"] = resampled["btc_dominance"].rolling(5).apply(calculate_slope, raw=False)
        resampled["volatility"] = resampled["btc_dominance"].rolling(10).std()

        latest = resampled.iloc[-1]
        features[f"btc_dom_{tf}"] = round(latest["btc_dominance"], 2)
        features[f"btc_dom_slope_{tf}"] = round(latest["slope"], 5)
        features[f"btc_dom_volatility_{tf}"] = round(latest["volatility"], 5)

    return features


def generate_btc_dominance_features(days: int = 30) -> dict:
    df = fetch_btc_dominance_history(days=days)
    return resample_and_generate_features(df)


if __name__ == "__main__":
    features = generate_btc_dominance_features()
    print("✅ BTC Dominance Features:")
    for k, v in features.items():
        print(f"{k}: {v}")
