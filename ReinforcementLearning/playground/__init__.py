from gymnasium.envs.registration import register
from copy import deepcopy
import sys
sys.path.append('/home/dz/')
from Stocks.TimeSeries.FetchData import fetch_stock_dataset


import pandas as pd
print("Registering the environment...")

STOCKS_GOOGL = pd.read_csv('/home/dz/Stocks/ReinforcementLearning/data/test/STOCKS_GOOGL.csv')
STOCKS_APPL = fetch_stock_dataset("AAPL")

register(
    id='stocks-v0',
    entry_point='ReinforcementLearning.playground.stocks_env:StocksEnv',
    kwargs={
        'df': deepcopy(STOCKS_APPL),
        'window_size': 30,
        'frame_bound': (30, len(STOCKS_APPL))
    }
)