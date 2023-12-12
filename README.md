# Stock-Price-Forecasting

## Project Overview

This project aims to trade stocks using Reinforcement Learning, by using a combination of various models and methods like Time Series Forecasting, Sentiment Analysis.
For demonstration purposes we are using the Google Stock Price Data, and News scraped from GoogleNews.

## Table of Contents

- [Architecture](#architecture)
- [Time Series Forecasting](#time-series-forecasting)
- [Sentiment Analysis](#sentiment-analysis)
- [Reinforcement Learning](#reinforcement-learning)
- [Results](#results)
- [References](#references)

## Architecture

![Architecture Diagram](https://github.com/Gugulugu/Stock-Price-Forecasting/blob/main/Documentation/Architecture.png)

### Time Series Forecasting

- **Model Used**: Transformer model
- **Data**: Google's stock price data
- **Purpose**: To capture and predict stock price trends over time

### Sentiment Analysis

- **Model Used**: FinancialBERT
- **Source**: [FinancialBERT Research Paper](https://www.researchgate.net/publication/358284785_FinancialBERT_-_A_Pretrained_Language_Model_for_Financial_Text_Mining)
- **Purpose**: To analyze the sentiment of financial news and its impact on stock prices

### Reinforcement Learning

- **Framework Used**: OpenAI Gym, Stable Baselines3
- **Approach**: Training an agent on a simulated stock market environment. The agent can only buy and sell shares and has to learn the best actions to maximise the profit.
- **Unique Aspect**: The agent is trained on both real and predicted stock prices, as well as news sentiment
- **Environment**: The environment is a Trading Environment [Gym-anytrading](https://github.com/AminHP/gym-anytrading) with only two actions BUY and SELL. We modified the environment so that the agent takes for every step the following information from the past (determined by the window_size) as input:

  - The current stock price
  - The predicted stock price
  - The sentiment of the news
  - The volume of the stock market

- **Datapreprocessing**: The Sentiment of the news is normalized and split into three features (positive, neutral, negative).
- **Evaluation**: I trained the agent on Google's share price from 2020 to 2021 with different algorithms to test and evaluate which RL algorithm performs best. Based on the metrics the Proximal Policy Optimisation (PPO) algorithm performed the best.

## Results

### Time Series Forecasting

![Time Series Forecasting Results](https://github.com/Gugulugu/Stock-Price-Forecasting/blob/main/Documentation/Prediction_Results.png)

### Sentiment Analysis

<img src="https://github.com/Gugulugu/Stock-Price-Forecasting/blob/main/Documentation/google_news_sentiment_class.png" width="500" height="300">

#### Baseline Model (Stock Price Only)

- ![Buy Sell Baseline](https://github.com/Gugulugu/Stock-Price-Forecasting/blob/main/Documentation/Buy_Sell_Baseline.png)
- ![Cumulative Returns Baseline](https://github.com/Gugulugu/Stock-Price-Forecasting/blob/main/Documentation/Cumulative_Returns_Test.png)

#### Enhanced Model (Stock Price, Predicted Price, Sentiment)

- ![Buy Sell Enhanced](https://github.com/Gugulugu/Stock-Price-Forecasting/blob/main/Documentation/Buy_Sell_Main.png)
- ![Cumulative Returns Enhanced](https://github.com/Gugulugu/Stock-Price-Forecasting/blob/main/Documentation/Cumulative_Returns_Main.png)

## References

- [FinancialBERT Research Paper](https://www.researchgate.net/publication/358284785_FinancialBERT_-_A_Pretrained_Language_Model_for_Financial_Text_Mining)
- [OpenAI Gym](https://gym.openai.com/)
- [Gym-anytrading](https://github.com/AminHP/gym-anytrading)
