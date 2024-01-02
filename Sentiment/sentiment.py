from transformers import pipeline
#from GoogleNewsCollector import get_news
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob

#get data
data_path = glob.glob("./Data/News/Merge/google_news_2012-01-01_to_2020-01-01.csv")
data_name = "google_train"
df_news = pd.read_csv(data_path[0])
print(df_news.shape)
print(df_news.head())
# load model
sentiment_pipeline = pipeline(model= "ahmedrachid/FinancialBERT-Sentiment-Analysis")

# get sentiment
data = df_news["description"].tolist()


print(sentiment_pipeline(data))

# get sentiment score
sentiment_output = sentiment_pipeline(data, truncation=True)
sentiment_score = [i['score'] for i in sentiment_output]
sentiment = [i['label'] for i in sentiment_output]
df_news["sentiment_score"] = sentiment_score
df_news["sentiment_class"] = sentiment
print(df_news.head())

# convert df to csv
df_news.to_csv("./Sentiment/Prediction_results/" + data_name + "_news_sentiment.csv", index=False)

# plot sentiment distribution

def plot_sentiment_distribution(df, column):
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column])
    plt.title(column + " Distribution")
    plt.savefig("./Sentiment/Prediction_results/"+ data_name + "_news_" + column + ".png")
    plt.show()


plot_sentiment_distribution(df_news, "sentiment_class")