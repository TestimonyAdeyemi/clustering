import streamlit as st
import requests
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN

# Function to gather images and text
def gather_images(url): 
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    images = []
    descriptions = []
    for img in soup.find_all("img"):
        src = img.get("src")
        try:
            image = Image.open(requests.get(src, stream=True).raw)
            images.append(image)
            descriptions.append(img.get("alt", ""))  # Use alt text as description
        except Exception as e:
            print(f"Error fetching image: {e}")
    return images, descriptions

# Function to extract image features (optional)
def extract_image_features(images):
    from keras.applications.vgg16 import VGG16, preprocess_input
    model = VGG16(weights='imagenet', include_top=False)
    features = []
    for image in images:
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        features.append(model.predict(image).flatten())
    return np.array(features)

# Function to cluster images
def cluster_images(data, clustering_algorithm):
    embedding = umap.UMAP().fit_transform(data)
    clusters = clustering_algorithm.fit_predict(embedding)
    return clusters

# Streamlit app structure
st.title("Product Image Clustering")

url = st.text_input("Enter website URL to crawl:")

if st.button("Gather Images"):
    images, descriptions = gather_images(url)

    cluster_by = st.radio("Cluster by:", ("Image Similarity", "Text Similarity"))

    if cluster_by == "Image Similarity":
        features = extract_image_features(images)
        clusters = cluster_images(features, DBSCAN())  # Replace with your chosen algorithm
    else:  # Text Similarity
        # Implement text-based clustering here
        pass

    # Display clusters
    for cluster_id in np.unique(clusters):
        st.header(f"Cluster {cluster_id}")
        for i, image in enumerate(images):
            if clusters[i] == cluster_id:
                st.image(image, caption=descriptions[i])
