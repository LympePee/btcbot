import requests
import pandas as pd
from datetime import datetime, timedelta

LOOKBACK_HOURS = 48

def get_btc_dominance_features():
    url = "https://api.coingecko.com/api/v3/global"
    resp = requests.get(url)
    data = resp.json()["data"]
    dominance = data["market_cap_percentage"]["btc"]

    # Για ιστορικό, μπορούμε να χρησιμοποιήσουμε /coins/bitcoin/market_chart
    chart = requests.get(
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
        params={"vs_currency": "usd", "days": LOOKBACK_HOURS / 24}
    ).json()

    df = pd.DataFrame(chart["market_caps"], columns=["ts", "cap"])
    df["dominance"] = df["cap"] / df["cap"].max() * dominance  # απλή εκτίμηση
    df["d_change"] = df["dominance"].pct_change().fillna(0) * 100

    return {
        "btc_d_avg_change": df["d_change"].mean(),
        "btc_d_volatility": df["d_change"].std(),
        "btc_d_change": df["d_change"].iloc[-1],
        "btc_d_current": dominance
    }
