import requests
import os
import json

# Load the config file
with open('config.json', 'r') as file:
    config = json.load(file)

url = "https://newsapi.org/v2/everything"
api_key = config['api_key']
query = "apple"

response = requests.get(url, params={
    "q": query,
    "apiKey": api_key,
    "pageSize": 5,  # Number of articles to retrieve
    "sortBy": "publishedAt",
    "language": "en"
})

data = response.json()
"""
for article in data.get("articles", []):
    print("Title:", article.get("title"))
    print("URL:", article.get("url"))
    print("Published At:", article.get("publishedAt"))
    print("Source:", article.get("source", {}).get("name"))
    print("Description:", article.get("description"))
    print("Content:", article.get("content"))
    print("-----\n")
"""

# data to df
import pandas as pd
df = pd.DataFrame(data['articles'])
print(df.head())