from transformers import pipeline
#from GoogleNewsCollector import get_news
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob

#get data
data_path = glob.glob("./Data/News/apple_news_*")
df_apple_news = pd.read_csv(data_path[0])
print(df_apple_news.shape)
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

# plot sentiment distribution

def plot_sentiment_distribution(df, column):
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column])
    plt.title(column + " Distribution")
    plt.savefig("./Sentiment/Prediction/apple_news_" + column + ".png")
    plt.show()


plot_sentiment_distribution(df_apple_news, "sentiment_class")