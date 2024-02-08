import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import datetime
import os
import json

LOKY_MAX_CPU_COUNT = 5

st.set_page_config(page_title="News")

# API endpoint to fetch news articles
news_api_url = "https://newsapi.org/v2/top-headlines?country=us&apiKey=8f41f07af3f244d58cc75ee2d238656b"

# Replace with your API key
YOUR_API_KEY = "8f41f07af3f244d58cc75ee2d238656b"

# Number of clusters
n_clusters = 5

# Function to fetch news articles and cache the response
def fetch_news():
    cache_file = "news_cache.json"
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
            if cache_data.get("date") == current_date:
                return cache_data["articles"]
    
    response = requests.get(news_api_url)
    articles = []
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        with open(cache_file, "w") as f:
            json.dump({"date": current_date, "articles": articles}, f)
    return articles

def main():
    st.title("News Article Clusters")

    # Fetch news articles
    articles = fetch_news()

    # Extract article titles and text content
    titles = [article["title"] for article in articles]
    contents = []
    for article in articles:
        content = article.get("description") or article.get("content")
        if content:
            contents.append(content)

    # Cluster articles
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(contents)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)

    # Display clusters
    for i in range(n_clusters):
        st.header(f"Cluster {i+1}")
        cluster_articles = [articles[j] for j, label in enumerate(kmeans.labels_) if label == i]
        for article in cluster_articles:
            st.markdown(f"* [{article['title']}]({article['url']})")

if __name__ == "__main__":
    main()
