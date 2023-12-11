# Stock-Price-Forecasting

## Project Overview

This project aims to predict future stock prices using various models and methods like Time Series Forecasting, Sentiment Analysis, and Reinforcement Learning. The primary focus is on Google's stock price, leveraging advanced models and frameworks.

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

- **Framework Used**: OpenAI Gym
- **Approach**: Training an agent on a simulated stock market environment
- **Unique Aspect**: The agent is trained on both real and predicted stock prices, as well as news sentiment

## Results

### Time Series Forecasting

![Time Series Forecasting Results](https://github.com/Gugulugu/Stock-Price-Forecasting/blob/main/Documentation/Prediction_Results.png)

### Sentiment Analysis

![Sentiment Analysis Results](https://github.com/Gugulugu/Stock-Price-Forecasting/blob/main/Documentation/google_news_sentiment_class.png)

### Reinforcement Learning

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
