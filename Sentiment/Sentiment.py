from transformers import pipeline
from GoogleNewsCollector import get_news

#get data
df_apple_news = get_news("Apple")
print(df_apple_news.head())
# load model
sentiment_pipeline = pipeline(model= "ahmedrachid/FinancialBERT-Sentiment-Analysis")

# get sentiment
data = df_apple_news["description"].tolist()


print(sentiment_pipeline(data))

# get sentiment score
sentiment_output = sentiment_pipeline(data, truncation=True)
sentiment_score = [i['score'] for i in sentiment_output]
sentiment = [i['label'] for i in sentiment_output]
df_apple_news["sentiment_score"] = sentiment_score
df_apple_news["sentiment_class"] = sentiment
print(df_apple_news.head())

# convert df to csv
df_apple_news.to_csv("./Sentiment/Prediction/apple_news_sentiment.csv", index=False)