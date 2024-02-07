#f29914eab5a34bd99550b4f9ca44056c

import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

LOKY_MAX_CPU_COUNT = 5

st.set_page_config(page_title="News")

# API endpoint to fetch news articles
news_api_url = "https://newsapi.org/v2/top-headlines?country=us&apiKey=f29914eab5a34bd99550b4f9ca44056c"

# Replace with your API key
YOUR_API_KEY = "f29914eab5a34bd99550b4f9ca44056c"

# Number of clusters
n_clusters = 5

def main():
  st.title("News Article Clusters")

  # Fetch news articles
  response = requests.get(news_api_url)
  articles = response.json()["articles"]

  # Extract article titles and text content
  titles = [article["title"] for article in articles]
  contents = []
  for article in articles:
    # Use Newspaper3k for cleaner extraction (optional)
    # url = article["url"]
    # article_text = Newspaper3k(url).article
    # contents.append(article_text)

    # Simple content extraction from description
    #contents.append(article["description"] or article["content"])
    contents = [article["description"] or article["content"] for article in articles if article["description"] or article["content"]]






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
