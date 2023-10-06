from gnews import GNews
import pandas as pd
import datetime



def get_news(keyword, start_date):
    delta = datetime.timedelta(days=1)
    today = datetime.date.today()
    df_news = pd.DataFrame()
    # iterate over range of dates
    while (start_date <= today):
        #print(start_date, end="\n")
        start_date_tuple = start_date.timetuple()[:3]
        end_date_tuple = (start_date + delta).timetuple()[:3]
        #print(start_date_tuple )
        #print(end_date_tuple )
        start_date += delta

        # get news
        google_news = GNews(
            language='en', 
            country='US', 
            max_results=10,
            start_date= start_date_tuple, 
            end_date= end_date_tuple,
            exclude_websites = ["9to5Toys.com", "9to5Google.com", "9to5Mac.com", "Electrek","ign.com"]
            )
        try:
            news = google_news.get_news(keyword)
            print("Found {} news for {} - {}".format(len(news), start_date_tuple, end_date_tuple))
            df_dict = pd.DataFrame(news)
            df_news = pd.concat([df_news, df_dict], ignore_index=True)
        except Exception as e:
            print(f"Failed to retrieve news for {start_date_tuple} - {end_date_tuple}: {e}")
    df_news = df_news.sort_values(by=['published date'], ascending=False)
    return df_news


df = get_news("Apple", datetime.date(2023,10,1))
print(df.head())
print(df[['title','published date']])
print(df.shape)
print(df.columns)