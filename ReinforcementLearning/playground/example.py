import sys
sys.path.append('/home/dz/Stocks/')
#from ReinforcementLearning.env import TradingEnv, StocksEnv, Actions, Positions 
from ReinforcementLearning.playground.stocks_env import StocksEnv  
from ReinforcementLearning.playground.trading_env import TradingEnv, Actions, Positions  

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym


#env = gym.make('forex-v0', frame_bound=(50, 100), window_size=10)
env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=10)

observation = env.reset(seed=2024)
while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # env.render()
    if done:
        print("info:", info)
        break

plt.cla()
env.unwrapped.render_all()
plt.show()