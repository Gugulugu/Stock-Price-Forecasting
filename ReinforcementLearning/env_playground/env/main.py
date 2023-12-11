from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stock_trading_env import StockTradingEnv
import matplotlib.pyplot as plt


# Create the environment
env = StockTradingEnv(csv_file='/home/dz/Stocks/ReinforcementLearning/data/test/STOCKS_GOOGL.csv')

# Vectorized environments allow for parallelism (optional but recommended for training efficiency)
vec_env = make_vec_env(lambda: env, n_envs=1)

# Initialize the model
#model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./tensorboard/")
model = DQN("MlpPolicy", vec_env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Test the trained model
state = env.reset()
portfolio_values = [] #

for i in range(100):
    action, _states = model.predict(state, deterministic=True)
    state, reward, done, info = env.step(action)
    #print("state:", state)
    #print("Step:", i, "Action:", action, "Reward:", reward, "Done:", done)
    portfolio_values.append(env.portfolio_value)  # Track portfolio value
    print(info)
    

    if done:
      break

# Optionally, save the model
#model.save("ppo_stock_trading")

# Print episode results
#env.plot_final_chart()

# Plot the portfolio values
plt.plot(portfolio_values)
plt.title("Portfolio Value Over Time")
plt.xlabel("Timesteps")
plt.ylabel("Portfolio Value")
plt.show()


# Close the environment
env.close()


