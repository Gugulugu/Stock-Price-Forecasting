import pandas as pd
import matplotlib.pyplot as plt

from trading_env import  Actions
from stocks_env import StocksEnv

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


import quantstats as qs
import numpy as np

#Train
df = pd.read_csv('./ReinforcementLearning/Dataset/Google_Sentiment_Forecast/Stock_Forecast_Dataset_train.csv')
df['Date'] = pd.to_datetime(df['Date'])
# date as index
df = df.set_index('Date')

#Test
df_test = pd.read_csv('./ReinforcementLearning/Dataset/Google_Sentiment_Forecast/Stock_Forecast_Dataset.csv')
df_test['Date'] = pd.to_datetime(df_test['Date'])
# date as index
df_test = df_test.set_index('Date')

timesteps = [500_000]

for timestep in timesteps:
    window_size = 30
    start_index = window_size
    end_index = len(df)
    learning_rate = 0.0003
    seed = 2023

    # check if columns contain non numeric values
    #print(df.applymap(np.isreal).all(1))
    #print(df.head())

    env = StocksEnv(
        df=df,
        window_size=window_size,
        frame_bound=(start_index, end_index)
    )

    def make_env():
        def _init():
            env = StocksEnv(
            df=df,
            window_size=window_size,
            frame_bound=(start_index, end_index)
        )  # Replace with your custom env
            return env
        return _init

    num_envs = 4  # Number of environments to run in parallel
    envs = [make_env() for _ in range(num_envs)]
    env = DummyVecEnv(envs)


    print("observation_space:", env.observation_space)

    #Train Env
    env.reset() #seed=2023
    
    model = PPO('MlpPolicy', env, verbose=0, tensorboard_log="./ReinforcementLearning/results/logs/",  learning_rate=learning_rate, seed=seed) #seed=99,
    print("Training...")
    model.learn(total_timesteps=timestep, progress_bar= True)
    env.close()


    test_window_size = window_size
    test_start_index = test_window_size
    test_end_index = len(df_test)

    test_env = StocksEnv(
        df=df_test,
        window_size=test_window_size,
        frame_bound=(test_start_index, test_end_index)
    )


    # Reset the test environment
    observation, info = test_env.reset(seed=seed)  # Using the same seed for consistency in comparison #seed=2023

    test_action_stats = {Actions.Sell: 0, Actions.Buy: 0, Actions.Hold: 0}

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

    def plot_results():
        # Plot Results
        plt.figure(figsize=(16, 6))
        test_env.unwrapped.render_all()
        plt.show()

    def plot_net_worth(test_env):
        # plot net worth
        plt.figure(figsize=(16, 6))
        plt.title("Net Worth")
        plt.xlabel("Step")
        plt.ylabel("Net Worth")
        plt.plot(test_env.unwrapped.history['total_profit'])
        plt.show()

    plot_results()
    plot_net_worth(test_env)



    # QuantStats
    qs.extend_pandas()

    net_worth = pd.Series(test_env.unwrapped.history['total_profit'], index=df_test.index[start_index+1:end_index])
    returns = net_worth.pct_change().iloc[1:]

    qs.reports.full(returns)
    qs.reports.html(returns, output='./ReinforcementLearning/results/quantstats/SB3_quantstats_ppo_test' + str(timestep) +'.html')
    qs.reports.metrics(returns)
    cumulative_returns = qs.stats.comp(returns)*100

    # add action stats and info to txt file
    f = open("./ReinforcementLearning/results/txt/output.txt", "a")
    f.write("Seed: " + str(seed) + "\n")
    f.write("Timestep: " + str(timestep) + "\n")
    f.write("Window Size: " + str(window_size) + "\n")
    f.write("Learning rate: " + str(learning_rate) + "\n")
    f.write("Test action stats: " + str(test_action_stats) + "\n")
    f.write("Test info: " + str(info) + "\n")
    f.write("Cumulative Returns: " + str(cumulative_returns) + "\n")
    f.close()