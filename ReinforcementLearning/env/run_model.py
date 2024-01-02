import pandas as pd
import matplotlib.pyplot as plt

from trading_env import  Actions
from stocks_env import StocksEnv

from stable_baselines3 import PPO,DQN, A2C

import quantstats as qs

# Create Env

df = pd.read_csv('./ReinforcementLearning/Dataset/Google_Sentiment_Forecast/Stock_Forecast_Dataset_train.csv')
df['Date'] = pd.to_datetime(df['Date'])
# date as index
df = df.set_index('Date')

print(df.head())

window_size = 10
start_index = window_size
end_index = len(df)

env = StocksEnv(
    df=df,
    window_size=window_size,
    frame_bound=(start_index, end_index)
)

print("observation_space:", env.observation_space)

# learning rate schedule
def learning_rate_schedule(progress):
    initial_lr = 0.003
    final_lr = 0.00003
    lr = initial_lr * (1 - progress) + final_lr * progress
    return lr

#Train Env
env.reset(seed=2023)
model = PPO('MlpPolicy', env, verbose=0, tensorboard_log="./ReinforcementLearning/results/logs/", learning_rate=learning_rate_schedule)
model.learn(total_timesteps=100_000, progress_bar= True)


#Test Env
action_stats = {Actions.Sell: 0, Actions.Buy: 0}

observation, info = env.reset(seed=2023)

while True:
    # action = env.action_space.sample()
    action, _states = model.predict(observation)

    action_stats[Actions(action)] += 1
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # env.render()
    if done:
        break

env.close()

print("action_stats:", action_stats)
print("info:", info)

# Plot Results
plt.figure(figsize=(16, 6))
env.unwrapped.render_all()
plt.show()

# plot net worth
plt.figure(figsize=(16, 6))
plt.title("Net Worth")
plt.xlabel("Step")
plt.ylabel("Net Worth")
plt.plot(env.unwrapped.history['total_profit'])
plt.show()

# QuantStats
qs.extend_pandas()

net_worth = pd.Series(env.unwrapped.history['total_profit'], index=df.index[start_index+1:end_index])
returns = net_worth.pct_change().iloc[1:]

qs.reports.full(returns)
qs.reports.html(returns, output='./ReinforcementLearning/results/SB3_a2c_quantstats.html')