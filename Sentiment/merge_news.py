import pandas as pd


df1 = pd.read_csv('./Data/News/2012-2020/google_news_2012-01-01_to_2015-05-13.csv')
df2 = pd.read_csv('./Data/News/2012-2020/google_news_2015-05-13_to_2016-04-29.csv')
df3 = pd.read_csv('./Data/News/2012-2020/google_news_2016-04-28_to_2020-01-02.csv')



combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# sort by date
combined_df = combined_df.sort_values(by=['date'], ascending=False)

last_date = '2020-01-01'
start_date = '2012-01-01'
#remove after 2021-01-01
combined_df = combined_df[combined_df['date'] < last_date]

# remove duplicates

combined_df.to_csv('./Data/News/Merge/google_news_' + start_date +'_to_' + last_date +'.csv', index=False)
