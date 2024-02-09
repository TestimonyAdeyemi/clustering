
This repository contains two Streamlit applications, each serving a different purpose. Below is an overview of each app along with instructions on how to run and contribute to the project.

## Unsplash Image Clustering

The Unsplash Image Clustering app fetches images from the Unsplash API based on user-provided queries, clusters them based on content or text descriptions (if available), and displays the images grouped by clusters. Users can specify the number of clusters and choose whether to cluster by image content or text descriptions.

### Code Structure

- **app_unsplash_image_clustering.py**: Main application file containing the Streamlit UI and logic for fetching images, clustering them, and displaying the clusters.

### Usage

1. Ensure you have the necessary dependencies installed:

   ```bash
   pip install streamlit requests scikit-learn pillow
   ```

2. Run the Streamlit application:

   ```bash
   streamlit run app_unsplash_image_clustering.py
   ```

3. The application will prompt you to enter a search query and select clustering options. After clicking the "Fetch and Cluster Images" button, it will fetch images, cluster them, and display the clusters in the browser.

### Contributing

Contributions to this app are welcome! If you'd like to contribute, please open an issue or submit a pull request.

## News Article Clusters

The News Article Clusters app fetches top news articles from the News API, clusters them based on their content, and displays the articles grouped by clusters. Each cluster represents a set of news articles with similar content.

### Code Structure

- **app_news_clusters.py**: Main application file containing the Streamlit UI and logic for fetching news articles, clustering them, and displaying the clusters.

### Usage

1. Ensure you have the necessary dependencies installed:

   ```bash
   pip install streamlit requests scikit-learn
   ```

2. Run the Streamlit application:

   ```bash
   streamlit run app_news_clusters.py
   ```

3. The application will fetch news articles, cluster them, and display the clusters in the browser.

### Contributing

Contributions to this app are welcome! If you'd like to contribute, please open an issue or submit a pull request.
