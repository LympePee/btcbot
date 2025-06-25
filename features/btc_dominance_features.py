# features/btc_dominance_features.py

import os
import pandas as pd
from datetime import datetime, timedelta
import ccxt

# === Settings ===
OUTPUT_CSV = "data/training_sets/btc_dominance_features.csv"
TIMEFRAME = "1h"
LOOKBACK_HOURS = 48
SYMBOLS = ["BTC/USDC", "ETH/USDC", "BNB/USDC", "XRP/USDC"]
DOMINANCE_SYMBOL = "BTC.D"
USDC_DOMINANCE_SYMBOL = "USDC.D"

# === Helper: fetch ohlcv from Binance ===
def fetch_ohlcv(symbol, since_ms):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since_ms)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# === Main Fetch Logic ===
now = datetime.utcnow()
since = int((now - timedelta(hours=LOOKBACK_HOURS)).timestamp() * 1000)

# Fetch BTC Dominance
btc_d = fetch_ohlcv(DOMINANCE_SYMBOL, since)
btc_d.set_index("timestamp", inplace=True)
btc_d = btc_d.resample("1h").last()

# Feature 1: BTC Dominance Change (4h)
btc_d["btc_dominance_change_4h"] = btc_d["close"].pct_change(periods=4)

# Feature 2-3: Price Changes for BTC & ETH
btc = fetch_ohlcv("BTC/USDC", since).set_index("timestamp").resample("1h").last()
eth = fetch_ohlcv("ETH/USDC", since).set_index("timestamp").resample("1h").last()
btc["btc_price_change_4h"] = btc["close"].pct_change(periods=4)
eth["eth_price_change_4h"] = eth["close"].pct_change(periods=4)

# Feature 4: Alt average return
altcoins = ["ETH/USDC", "BNB/USDC", "XRP/USDC"]
alt_df = pd.DataFrame()

for symbol in altcoins:
    alt = fetch_ohlcv(symbol, since).set_index("timestamp").resample("1h").last()
    alt[f"{symbol}_return"] = alt["close"].pct_change(periods=4)
    alt_df = pd.concat([alt_df, alt[[f"{symbol}_return"]]], axis=1)

alt_df["alt_avg_return_4h"] = alt_df.mean(axis=1)

# Feature 5: Market Breadth (1h)
market_breadth = (alt_df > 0).sum(axis=1) / len(altcoins)
market_breadth = market_breadth.rename("market_breadth_1h")

# Feature 6: Correlation BTC vs ETH/BNB/XRP
cor_df = pd.concat([
    btc["close"].pct_change(),
    fetch_ohlcv("BNB/USDC", since).set_index("timestamp")["close"].pct_change(),
    fetch_ohlcv("XRP/USDC", since).set_index("timestamp")["close"].pct_change(),
    eth["close"].pct_change()
], axis=1)

cor_df.columns = ["btc", "bnb", "xrp", "eth"]
correlation = cor_df.rolling(window=4).corr().loc[:, "btc"].drop(columns="btc")
correlation["correlation_btc_vs_eth_bnb_xrp"] = correlation.mean(axis=1)

# Feature 7: BTC Volume Dominance
btc_vol = btc["volume"]
total_vol = sum([fetch_ohlcv(s, since).set_index("timestamp")["volume"] for s in SYMBOLS])
btc_volume_dominance = (btc_vol / total_vol).rename("btc_volume_dominance")

# Feature 8: USDC Dominance Change
usdc_d = fetch_ohlcv(USDC_DOMINANCE_SYMBOL, since).set_index("timestamp").resample("1h").last()
usdc_d["usdc_d_change"] = usdc_d["close"].pct_change(periods=4)

# === Combine all features ===
final = pd.concat([
    btc_d["btc_dominance_change_4h"],
    btc["btc_price_change_4h"],
    eth["eth_price_change_4h"],
    alt_df["alt_avg_return_4h"],
    market_breadth,
    correlation["correlation_btc_vs_eth_bnb_xrp"],
    btc_volume_dominance,
    usdc_d["usdc_d_change"]
], axis=1).dropna()

# === Save to CSV ===
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
final.to_csv(OUTPUT_CSV)
print(f"✅ BTC Dominance features saved to {OUTPUT_CSV}")
