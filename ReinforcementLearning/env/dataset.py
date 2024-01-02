import pandas as pd
import numpy as np

# Preprocess the data from sentiment analysis and forecasting

forecast_file_path = "./Data/Forecasting/Merge/google_forecasting_2012-01-01_to_2020-01-01.csv"
sentiment_file_path ="./Sentiment/Prediction_results/google_train_news_sentiment.csv"

# Load the forecast data
df_forecast = pd.read_csv(forecast_file_path)
# Load the sentiment data
df_sentiment = pd.read_csv(sentiment_file_path)

# Preprocess forecast data
def preprocess_forecast_data(df_forecast, start_date, end_date):
    # Convert Date column to datetime
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

    #remove columns Day,Year,Month
    df_forecast.drop(['Day','Year','Month'],axis=1,inplace=True)

    # volume to log
    #df_forecast['Volume'] = df_forecast['Volume'].apply(lambda x: np.log(x))

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df_filtered = df_forecast[(df_forecast['Date'] >= start_date) & (df_forecast['Date'] <= end_date)]


    return df_filtered

# Preprocess sentiment data
def preprocess_sentiment_data(df_sentiment):
    # Group by date and sentiment class and counting occurrences
    df_sentiment_grouped = df_sentiment.groupby(['date', 'sentiment_class']).size().unstack(fill_value=0)

    df_sentiment_grouped.columns = [col for col in df_sentiment_grouped.columns]

    df_sentiment_grouped.reset_index(inplace=True)

    df_sentiment_grouped['neutral_norm'] = df_sentiment_grouped['neutral'] / (df_sentiment_grouped['neutral'] + df_sentiment_grouped['negative'] + df_sentiment_grouped['positive']) 
    df_sentiment_grouped['negative_norm'] = df_sentiment_grouped['negative'] / (df_sentiment_grouped['neutral'] + df_sentiment_grouped['negative'] + df_sentiment_grouped['positive']) 
    df_sentiment_grouped['positive_norm'] = df_sentiment_grouped['positive'] / (df_sentiment_grouped['neutral'] + df_sentiment_grouped['negative'] + df_sentiment_grouped['positive']) 

    df_sentiment_grouped.drop(['neutral','negative','positive'],axis=1,inplace=True)
    df_sentiment_grouped['date'] = pd.to_datetime(df_sentiment_grouped['date'])


    return df_sentiment_grouped

# Merge the two dataframes
def merge_dataframes(df_forecast, df_sentiment):
    df_merged = pd.merge(df_forecast, df_sentiment, how='inner', left_on='Date', right_on='date')

    df_merged.drop(['date'],axis=1,inplace=True)
    # sort by date
    df_merged.sort_values(by=['Date'], inplace=True)

    return df_merged

# Preprocess the data
df_forecast = preprocess_forecast_data(df_forecast, start_date='2012-01-01', end_date='2020-01-01' )
df_sentiment = preprocess_sentiment_data(df_sentiment)
df_merged = merge_dataframes(df_forecast, df_sentiment)
print(df_merged.head())

# Save the merged dataframe
df_merged.to_csv("./ReinforcementLearning/Dataset/Google_Sentiment_Forecast/Stock_Forecast_Dataset_train.csv", index=False)
