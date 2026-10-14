import yfinance as yf
import pandas as pd
from datetime import datetime

WATCHLIST = [“SOUN”, “AAPL”, “PLTR”, “BBAI”, “OPEN”]

def analyze_stock(symbol):
try:
stock = yf.Ticker(symbol)
data = stock.history(period=“3mo”, interval=“1d”)

    if data.empty or len(data) < 20:
        return None
    close = data["Close"]
    volume = data["Volume"]
    price = float(close.iloc[-1])
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    change = ((price - float(close.iloc[-2])) /
              float(close.iloc[-2])) * 100
    avg_volume = float(volume.tail(20).mean())
    last_volume = float(volume.iloc[-1])
    return {
        "symbol": symbol,
        "price": price,
        "ema20": ema20,
        "sma20": sma20,
        "change": change,
        "avg_volume": avg_volume,
        "last_volume": last_volume,
    }
except Exception as error:
    print(f"Error analyzing {symbol}: {error}")
    return None

def main():
print(“US STOCK SCANNER”)
print(“Scan time:”, datetime.now().strftime(”%Y-%m-%d %H:%M”))
print(”-” * 40)

for symbol in WATCHLIST:
    result = analyze_stock(symbol)
    if result is None:
        continue
    near_ema = (
        abs(result["price"] - result["ema20"]) /
        result["ema20"] * 100
    ) <= 3
    print(f"\nSymbol: {result['symbol']}")
    print(f"Price: ${result['price']:.2f}")
    print(f"Daily change: {result['change']:.2f}%")
    print(f"EMA20: ${result['ema20']:.2f}")
    print(f"SMA20: ${result['sma20']:.2f}")
    print(f"Average volume: {result['avg_volume']:.0f}")
    print(f"Last volume: {result['last_volume']:.0f}")
    if near_ema:
        print("Signal: Near EMA20 - needs further review")

if name == “main”:
main()
