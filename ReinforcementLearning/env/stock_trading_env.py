from typing import Tuple
import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


STOCKS_GOOGL = pd.read_csv('/home/dz/Stocks/ReinforcementLearning/data/test/STOCKS_GOOGL.csv')


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    def __init__(self, csv_file, initial_capital=10000, max_buy=10, max_sell=10, transaction_fee=0):
        # Load the dataset
        self.stock_data = pd.read_csv(csv_file)
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])

        # Set initial capital and stock holding
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.stock_holding = 0

        # Set transaction parameters
        self.max_buy = max_buy
        self.max_sell = max_sell
        self.transaction_fee = transaction_fee

        # Initialize state
        self.current_step = 0
        self.done = False

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3) # 3 * (self.max_buy + self.max_sell + 1)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(6,))
        
        # Initialize action history
        self.action_history = []

        # Initialize action log
        self.action_log = pd.DataFrame(columns=['Step', 'Price', 'Action'])


        # Additional environment settings
        self.window_size = 10
        self.seed(42)  # Optional for reproducibility


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


    def step(self, action):
        # Current stock price
        current_price = self.stock_data.loc[self.current_step, "Close"]
        action_type = 'Hold' if action == 0 else ('Buy' if action == 1 else 'Sell')
        self.action_log.loc[len(self.action_log)] = [self.current_step, current_price, action_type]


        # Update action history
        self.action_history.append(action)

        if action == 1:  # Buy
            # Calculate the number of shares to buy
            num_shares = min(self.capital // current_price, self.max_buy)
            self.stock_holding += num_shares
            self.capital -= num_shares * current_price

        elif action == 2:  # Sell
            # Calculate the number of shares to sell
            num_shares = min(self.stock_holding, self.max_sell)
            self.stock_holding -= num_shares
            self.capital += num_shares * current_price

        # Calculate the reward
        new_portfolio_value = self.stock_holding * current_price + self.capital
        reward = new_portfolio_value - self.initial_capital

        # Update the state to the next day
        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 1

        info = {}

        return self._next_observation(), reward, done, info

            


    def render(self, mode='human', close=False):
        if mode != 'human':
            raise NotImplementedError("Supported render modes: human")

        if close:
            plt.close()
            return
        

        # Plot the stock price chart
        plt.figure(figsize=(10, 6))
        plt.plot(self.stock_data['Date'], self.stock_data['Close'], label='Close Price')

        # Plot buy and sell actions
        buys = self.action_log[self.action_log['Action'] == 'Buy']
        sells = self.action_log[self.action_log['Action'] == 'Sell']
        plt.scatter(self.stock_data['Date'].iloc[buys['Step']], buys['Price'], color='green', label='Buy', marker='^', alpha=0.7)
        plt.scatter(self.stock_data['Date'].iloc[sells['Step']], sells['Price'], color='red', label='Sell', marker='v', alpha=0.7)

        plt.title("Stock Price Chart with Actions")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()

        plt.show()



    def reset(self):
        # Reset capital and stock holding to their initial states
        self.capital = self.initial_capital
        self.stock_holding = 0

        # Reset the action history
        self.action_history = []


        # Reset the step to the beginning
        self.current_step = 0

        # Return the initial observation
        return self._next_observation()
    
    def _next_observation(self):
        # Get the data for the current step
        frame = self.stock_data.iloc[self.current_step]

        # Include price data and portfolio information
        obs = np.array([frame['Open'], frame['High'], frame['Low'], frame['Close'], frame['Volume'], self.capital])

        return obs
    
    def plot_final_chart(self):
        # Plot the stock price chart
        plt.figure(figsize=(10, 6))
        plt.plot(self.stock_data['Date'], self.stock_data['Close'], label='Close Price')

        # Plot buy and sell actions
        buys = self.action_log[self.action_log['Action'] == 'Buy']
        sells = self.action_log[self.action_log['Action'] == 'Sell']
        plt.scatter(self.stock_data['Date'].iloc[buys['Step']], buys['Price'], color='green', label='Buy', marker='^', alpha=0.7)
        plt.scatter(self.stock_data['Date'].iloc[sells['Step']], sells['Price'], color='red', label='Sell', marker='v', alpha=0.7)

        plt.title("Stock Price Chart with Actions")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.show()    


