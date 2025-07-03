from btcbot.utils.fetch_binance import fetch_ohlcv_binance, save_ohlcv_to_csv as save_binance
from btcbot.utils.fetch_kraken import fetch_ohlcv_kraken, save_ohlcv_to_csv as save_kraken
import ccxt


def fetch_and_save_ohlcv(exchange: str, symbol: str, timeframe: str, start_date: str):
    """
    General-purpose wrapper to fetch and save OHLCV data from supported exchanges.
    """
    since = ccxt.binance().parse8601(start_date)

    if exchange.lower() == "binance":
        df = fetch_ohlcv_binance(symbol, timeframe, since)
        save_binance(df, symbol, timeframe)
    elif exchange.lower() == "kraken":
        df = fetch_ohlcv_kraken(symbol, timeframe, since)
        save_kraken(df, symbol, timeframe)
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")


if __name__ == "__main__":
    # Example usage
    fetch_and_save_ohlcv(
        exchange="binance",
        symbol="BTC/USDC",
        timeframe="1h",
        start_date="2024-01-01T00:00:00Z"
    )
