import numpy as np
from collections import Counter
from trading_env import TradingEnv, Actions, Positions


class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound

        super().__init__(df, window_size, render_mode)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    def _process_data(self):
        prices_data = self.df.loc[:, 'Close'].to_numpy()
        predicted_price = self.df.loc[:, 'Predicted_Close'].to_numpy()
        volume = self.df.loc[:, 'Volume'].to_numpy()
        neutral_norm = self.df.loc[:, 'neutral_norm'].to_numpy()
        negative_norm = self.df.loc[:, 'negative_norm'].to_numpy()
        positive_norm = self.df.loc[:, 'positive_norm'].to_numpy()

        # normalize data
        #prices = StocksEnv.min_max_normalize(prices)
        #predicted_price = StocksEnv.min_max_normalize(predicted_price)
        volume = StocksEnv.min_max_normalize(volume)



        prices_data[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices_data[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        predicted_price = predicted_price[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        neutral_norm = neutral_norm[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        negative_norm = negative_norm[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        positive_norm = positive_norm[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        volume = volume[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        # scale columns to mean price values
        volume = StocksEnv.scale_based_on_price(volume, prices)
        neutral_norm = StocksEnv.scale_based_on_price(neutral_norm, prices)
        negative_norm = StocksEnv.scale_based_on_price(negative_norm, prices)
        positive_norm = StocksEnv.scale_based_on_price(positive_norm, prices)





        



        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff, positive_norm, negative_norm, neutral_norm, volume ,predicted_price))
        #print(signal_features)

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action):

        action_history_counter = Counter(self._action_history)
        sell_count = action_history_counter[0]
        buy_count = action_history_counter[1]
        hold_count = action_history_counter[2]
        
        step_reward = 0

        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True
        elif action == Actions.Hold.value:  # New condition for Hold action
            trade = False

        if trade:
            # add negative reward for buying too much
            current_price = self.prices[self._current_tick] * (1 + self.trade_fee_ask_percent) # ask price
            last_trade_price = self.prices[self._last_trade_tick] * (1 + self.trade_fee_bid_percent) # bid price
            price_diff = current_price - last_trade_price
        # implement if more share bought than sold then reward is negative
            if sell_count > buy_count:
                step_reward += price_diff
            elif sell_count < buy_count:
                step_reward -= price_diff
            else:
                step_reward += price_diff

            """
            if self._position == Positions.Long:
                step_reward += price_diff
            """
        return step_reward
        

    
    def _update_profit(self, action):
        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True
        
        elif action == Actions.Hold.value:  # New condition for Hold action
            trade = False

        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                if last_trade_price == 0:
                    last_trade_price = 0.0001
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
    
    def min_max_normalize(column):
        return (column - column.min()) / (column.max() - column.min())
    
    def scale_based_on_price(column, prices):
        return column * prices.mean()
    
