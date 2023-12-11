import pandas as pd
import numpy as np

# Preprocess the data from sentiment analysis and forecasting

forecast_file_path = "./Data/Forecasting/Prediction_Results.csv"

# Load the forecast data
df_forecast = pd.read_csv(forecast_file_path)


# Preprocess forecast data
def preprocess_data(df_forecast, start_date, end_date):
    # Convert Date column to datetime
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

    #remove columns Day,Year,Month
    df_forecast.drop(['Day','Year','Month', 'Predicted_Close'],axis=1,inplace=True)

    # volume to log
    #df_forecast['Volume'] = df_forecast['Volume'].apply(lambda x: np.log(x))

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df_filtered = df_forecast[(df_forecast['Date'] >= start_date) & (df_forecast['Date'] <= end_date)]


    return df_filtered

# Preprocess the data
df = preprocess_data(df_forecast, start_date='2020-01-01', end_date='2021-01-01' )
print(df.head())

# Save the merged dataframe
df.to_csv("./ReinforcementLearning/Dataset/test/Stock_Dataset.csv", index=False)
