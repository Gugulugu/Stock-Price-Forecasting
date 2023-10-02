from transformers import pipeline
sentiment_pipeline = pipeline(model= "ahmedrachid/FinancialBERT-Sentiment-Analysis")
data = ["I love you", "I hate you"]
#sentiment_pipeline(data)
print(sentiment_pipeline(data))