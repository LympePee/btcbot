# btcbot/utils/fetch_binance_save_csv.py

import os
from btcbot.utils.fetch_binance import fetch_ohlcv_binance  # ✅ Absolute import μέσα στο btcbot

def fetch_binance_ohlcv(symbol="BTC/USDC", timeframe="1h", since=1483228800000):
    df = fetch_ohlcv_binance(symbol, timeframe, since)
    
    # Προαιρετικά μπορείς να ορίσεις entry_price
    # df["entry_price"] = df["close"]
    
    out_dir = "btcbot/data/historical/binance/BTCUSDC"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{timeframe}.csv")
    
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {symbol} {timeframe} to {out_path}")
    
    return df
