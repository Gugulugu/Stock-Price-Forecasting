import pandas as pd


df1 = pd.read_csv('./Data/Forecasting/2012-2020/Prediction_Results_2011-12-12-2013-01-01.csv')
df2 = pd.read_csv('./Data/Forecasting/2012-2020/Prediction_Results_2012-12-12-2014-01-01.csv')
df3 = pd.read_csv('./Data/Forecasting/2012-2020/Prediction_Results_2013-12-12-2015-01-01.csv')
df4 = pd.read_csv('./Data/Forecasting/2012-2020/Prediction_Results_2014-12-12-2016-01-01.csv')
df5 = pd.read_csv('./Data/Forecasting/2012-2020/Prediction_Results_2015-12-12-2017-01-01.csv')
df6 = pd.read_csv('./Data/Forecasting/2012-2020/Prediction_Results_2016-12-12-2018-01-01.csv')
df7 = pd.read_csv('./Data/Forecasting/2012-2020/Prediction_Results_2017-12-12-2019-01-01.csv')
df8 = pd.read_csv('./Data/Forecasting/2012-2020/Prediction_Results_2018-12-12-2020-01-01.csv')

# remove rows before date
df1 = df1[df1['Date'] >= '2012-01-01']
df2 = df2[df2['Date'] >= '2013-01-01']
df3 = df3[df3['Date'] >= '2014-01-01']
df4 = df4[df4['Date'] >= '2015-01-01']
df5 = df5[df5['Date'] >= '2016-01-01']
df6 = df6[df6['Date'] >= '2017-01-01']
df7 = df7[df7['Date'] >= '2018-01-01']
df8 = df8[df8['Date'] >= '2019-01-01']


start_date = "2012-01-01"
last_date = "2020-01-01"

combined_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)

# sort by date
combined_df = combined_df.sort_values(by=['Date'], ascending=False)

combined_df.to_csv('./Data/Forecasting/Merge/google_forecasting_' + start_date +'_to_' + last_date +'.csv', index=False)
