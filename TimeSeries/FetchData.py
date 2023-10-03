import yfinance as yf
import pandas as pd
from datetime import date

def fetch_stock_dataset(ticker_symbol):
    # Define the ticker symbol for the NASDAQ Composite index
    # ticker_symbol = "AAPL"
    # get date of today
    today = date.today()

    # Fetch the data
    data = yf.download(ticker_symbol, start="2000-01-01", end=today)

    # Data is already in a DataFrame, so you can directly use it
    # print(data.head())
    df_stock_dataset = data.reset_index()
    return df_stock_dataset

df = fetch_stock_dataset("AAPL")
print(df.head())
