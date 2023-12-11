import pandas as pd


df1 = pd.read_csv('/home/dz/Stocks/Data/News/google_news_2020-01-01_to_2020-03-19.csv')
df2 = pd.read_csv('/home/dz/Stocks/Data/News/google_news_2020-03-19_to_2020-06-03.csv')
df3 = pd.read_csv('/home/dz/Stocks/Data/News/google_news_2020-06-03_to_2020-07-11.csv')
df4 = pd.read_csv('/home/dz/Stocks/Data/News/google_news_2020-07-10_to_2020-09-13.csv')
df5 = pd.read_csv('/home/dz/Stocks/Data/News/google_news_2020-09-13_to_2020-10-24.csv')
df6 = pd.read_csv('/home/dz/Stocks/Data/News/google_news_2020-10-24_to_2021-01-02.csv')


combined_df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

# sort by date
combined_df = combined_df.sort_values(by=['date'], ascending=False)


combined_df.to_csv('/home/dz/Stocks/Data/News/google_news_2020-01-01_to_2021-01-01.csv', index=False)
