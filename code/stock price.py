import yfinance as yf
import pandas as pd

tickers = ["TSLA", "NIO", "RIVN", "AAPL", "MSFT", "GOOG", "NVDA", "AMZN", "META"]

# auto_adjust=True 会把复权价放进 'Close'
raw = yf.download(tickers, start="2024-01-01", end="2024-12-31",
                  interval="1d", auto_adjust=True, progress=False)

# 兼容单票/多票两种列结构，拿到收盘价
if isinstance(raw.columns, pd.MultiIndex):
    data = raw["Close"]            # 多票：顶层字段 -> 'Close'
else:
    data = raw[["Close"]]          # 单票：普通列

train = data.loc["2024-01-01":"2024-09-30"]
test  = data.loc["2024-10-01":"2024-12-31"]

train.to_csv("train_stock_data.csv")
test.to_csv("test_stock_data.csv")
print(train.head())
