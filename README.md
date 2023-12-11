# Stock-Price-Forecasting

Goal of this project is to create a model to predict the future stock price, based on different kind of information and models (Time Series Forecasting, Sentiment Analysis, ...)

## Architecture:

![results](https://github.com/Gugulugu/Stock-Price-Forecasting/blob/main/Documentation/Architecture.png)

## Time Series Forecasting

For Time Series forecasting I trained a transformer model on stock price data from google

### Results:

![results](https://github.com/Gugulugu/Stock-Price-Forecasting/blob/main/Documentation/Prediction_Results.png)

## Sentiment Analysis

For Sentiment analysis a Pretrained Transformer-Model [FinancialBERT](https://www.researchgate.net/publication/358284785_FinancialBERT_-_A_Pretrained_Language_Model_for_Financial_Text_Mining) was used

### Results:

![results](https://github.com/Gugulugu/Stock-Price-Forecasting/blob/main/Documentation/google_news_sentiment_class.png)

## Reinforcement Learning

The reinforcement learning part is built using the OpenAI Gym framework. The environment is a stock market with a stock price that changes over time. The agent can only buy and sell shares and has to learn the best actions to maximise the profit. The special approach in this project is that the agent is trained not only on the stock price ("close") itself, but also on the sentiment of the news and the predicted stock price from the stock price forecasting model. The sentiment is calculated using the FinancialBERT model.

### Datapreprocessing

- The Sentiment of the news is normalized and split into three features (positive, neutral, negative).
- The real stock price + the predicted stock price + the sentiment scores are merged into one dataframe.

### Environment

The environment is a Trading Environment with only two actions BUY and SELL. The agent takes for every step the following information from the past as input:

- The current stock price
- The predicted stock price
- The sentiment of the news
- The volume of the stock market

How far back in time is determined by the Window Size. The Window Size is the number of days that are taken into account for every step. The Window Size is a hyperparameter and can be changed.

### Model Evaluation and Testing

We trained the agent on Google's share price from 2020 to 2021 with different algorithms to test and evaluate which RL algorithm performs best. As a result, we found that the Proximal Policy Optimisation (PPO) algorithm performed best.

### Results

#### Stock Price Prediction only using the Stock Price as input:

![results](![Alt text](https://github.com/Gugulugu/Stock-Price-Forecasting/blob/main/Documentation/Buy_Sell_Baseline.png))
![results](![Alt text](https://github.com/Gugulugu/Stock-Price-Forecasting/blob/main/Documentation/Cumulative_Returns_Test.png))

#### Stock Price Prediction using the predicted future Stock Price and the Sentiment as input:

![results](![Alt text](https://github.com/Gugulugu/Stock-Price-Forecasting/blob/main/Documentation/Buy_Sell_Main.png))
![results](![Alt text](https://github.com/Gugulugu/Stock-Price-Forecasting/blob/main/Documentation/Cumulative_Returns_Main.png))
