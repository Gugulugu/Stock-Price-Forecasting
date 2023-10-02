from gnews import GNews
import pandas as pd

def get_news(keyword):
    google_news = GNews(exclude_websites=["9to5Toys.com", "9to5Google.com", "9to5Mac.com", "Electrek","ign.com"])
    news = google_news.get_news(keyword)
    df_news = pd.DataFrame(news)
    return df_news
