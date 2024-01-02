import yfinance as yf
import pandas as pd
from datetime import date

def fetch_stock_dataset(ticker_symbol, start_date = "2000-01-01", end_date = date.today()):
    # Define the ticker symbol for the NASDAQ Composite index
    # ticker_symbol = "AAPL"
    # get date of today

    # Fetch the data
    data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # Data is already in a DataFrame, so you can directly use it
    # print(data.head())
    df_stock_dataset = data.reset_index()
    return df_stock_dataset

df = fetch_stock_dataset("AAPL")
print(df.head())
