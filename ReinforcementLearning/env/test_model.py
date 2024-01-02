import pandas as pd
import matplotlib.pyplot as plt

from trading_env import  Actions
from stocks_env import StocksEnv

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


import quantstats as qs
import numpy as np

# logger
new_logger = configure("./ReinforcementLearning/results/logs/", ["stdout", "tensorboard"])
Logger.DEFAULT = new_logger


# Define a learning rate schedule
def learning_rate_schedule(progress):
    initial_lr = 0.0003
    final_lr = 0.000003
    lr = initial_lr * (1 - progress) + final_lr * progress
    return lr

# Create Env

df = pd.read_csv('./ReinforcementLearning/Dataset/Google_Sentiment_Forecast/Stock_Forecast_Dataset_train.csv')

df['Date'] = pd.to_datetime(df['Date'])
# date as index
df = df.set_index('Date')

window_size = 10
start_index = window_size
end_index = len(df)

# check if columns contain non numeric values
#print(df.applymap(np.isreal).all(1))
print(df.head())

env = StocksEnv(
    df=df,
    window_size=window_size,
    frame_bound=(start_index, end_index)
)
"""
def make_env(df, window_size, start_index, end_index, rank):
    def _init():
        env = StocksEnv(df=df, window_size=window_size, frame_bound=(start_index, end_index))
        env.seed(2023 + rank)
        return env
    return _init

num_envs = 2  # Number of environments to run in parallel
envs = [make_env(df, window_size, start_index, end_index, i) for i in range(num_envs)]
env = SubprocVecEnv(envs)
"""
print("observation_space:", env.observation_space)

#Train Env
env.reset(seed=2000)
model = A2C('MlpPolicy', env, verbose=0, tensorboard_log="./ReinforcementLearning/results/logs/", learning_rate=learning_rate_schedule)
print("Training...")
model.learn(total_timesteps=10_000, progress_bar= True)
env.close()

df_test = pd.read_csv('./ReinforcementLearning/Dataset/Google_Sentiment_Forecast/Stock_Forecast_Dataset.csv')
df_test['Date'] = pd.to_datetime(df_test['Date'])
# date as index
df_test = df_test.set_index('Date')

test_window_size = window_size
test_start_index = test_window_size
test_end_index = len(df_test)

test_env = StocksEnv(
    df=df_test,
    window_size=test_window_size,
    frame_bound=(test_start_index, test_end_index)
)


# Reset the test environment
observation, info = test_env.reset(seed=2000)  # Using the same seed for consistency in comparison

test_action_stats = {Actions.Sell: 0, Actions.Buy: 0}

print("Testing...")
while True:
    action, _states = model.predict(observation)
    test_action_stats[Actions(action)] += 1
    observation, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated

    if done:
        break

test_env.close()

print("Test action stats:", test_action_stats)
print("Test info:", info)

# Plot Results
plt.figure(figsize=(16, 6))
test_env.unwrapped.render_all()
plt.show()

# plot net worth
plt.figure(figsize=(16, 6))
plt.title("Net Worth")
plt.xlabel("Step")
plt.ylabel("Net Worth")
plt.plot(test_env.unwrapped.history['total_profit'])
plt.show()

# QuantStats
qs.extend_pandas()

net_worth = pd.Series(test_env.unwrapped.history['total_profit'], index=df_test.index[start_index+1:end_index])
returns = net_worth.pct_change().iloc[1:]

qs.reports.full(returns)
qs.reports.html(returns, output='./ReinforcementLearning/results/SB3_a2c_quantstats.html')