from gnews import GNews
import pandas as pd
import datetime
import requests





def create_news_dataset(keyword, start_date, end_date):
    start = start_date
    delta = datetime.timedelta(days=1)
    today = datetime.date.today()


    if end_date is None:
        end_date = today
    else:
        today = end_date

    df_news = pd.DataFrame()
    # iterate over range of dates
    while (start_date <= today):
        #print(start_date, end="\n")
        start_date_tuple = start_date.timetuple()[:3]
        current_end_date = start_date + delta
        end_date_tuple = (start_date + delta).timetuple()[:3]

        #print(start_date_tuple )
        #print(end_date_tuple )
        start_date += delta

        # get news
        google_news = GNews(
            language='en', 
            country='US', 
            max_results=100,
            start_date= start_date_tuple, 
            end_date= end_date_tuple,
            exclude_websites = ["9to5Toys.com", "9to5Google.com", "9to5Mac.com", "Electrek","ign.com"],
            #proxy= proxy_dict,
            )
        try:
            news = google_news.get_news(keyword)
            print("Found {} news for {} - {}".format(len(news), start_date_tuple, end_date_tuple))
            if len(news) == 0:
                break
            df_dict = pd.DataFrame(news)
            df_dict['date'] = start_date - delta
            df_news = pd.concat([df_news, df_dict], ignore_index=True)
            # add date
            #df_news['date'] = start_date_tuple
        except Exception as e:
            print(f"Failed to retrieve news for {start_date_tuple} - {end_date_tuple}: {e}")

    try:
        
        df_news = df_news.sort_values(by=['date'], ascending=False)

        df_news.to_csv('./Data/News/' + search_term +'_news_' + str(start) + '_to_' + str(current_end_date) +'.csv', index=False)
    except Exception as e:
        print(f"Error: IP has been blocked!")

    return df_news

search_term = "google" 
df = create_news_dataset(search_term, datetime.date(2020,10,24), datetime.date(2021,1,1))
print(df.head())
#print(df[['title','published date']])
print(df.shape)
print(df.columns)
