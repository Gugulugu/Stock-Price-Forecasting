import sys
sys.path.append('/home/dz/Stocks/')
from ReinforcementLearning.env.stocks_env import StocksEnv  # Add this line

import gymnasium as gym
#import stocks_env
#import trading_env
import pandas as pd

STOCKS_GOOGL = pd.read_csv('/home/dz/Stocks/ReinforcementLearning/data/test/STOCKS_GOOGL.csv')


custom_env = gym.make(
    'stocks-v0',
    df=STOCKS_GOOGL,
    window_size=10,
    frame_bound=(10, 300)
)
"""
print("env information:")
print("> shape:", env.unwrapped.shape)
print("> df.shape:", env.unwrapped.df.shape)
print("> prices.shape:", env.unwrapped.prices.shape)
print("> signal_features.shape:", env.unwrapped.signal_features.shape)
print("> max_possible_profit:", env.unwrapped.max_possible_profit())
"""
print()
print("custom_env information:")
print("> shape:", custom_env.unwrapped.shape)
print("> df.shape:", custom_env.unwrapped.df.shape)
print("> prices.shape:", custom_env.unwrapped.prices.shape)
print("> signal_features.shape:", custom_env.unwrapped.signal_features.shape)
print("> max_possible_profit:", custom_env.unwrapped.max_possible_profit())

custom_env.reset()
custom_env.render()
custom_env.pause_rendering()