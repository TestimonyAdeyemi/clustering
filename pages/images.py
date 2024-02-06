import streamlit as st
import requests
from PIL import Image
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN
import unsplash

# Set your Unsplash access key
unsplash.set_api_key("MFcvfNHZ-u_kRAhK4oHT2uIIYtWVTxI-pk8qbr4sIR8")

# Function to gather images from Unsplash
def gather_images(query):
    photos = unsplash.search.photos(query=query, per_page=30)  # Adjust per_page as needed
    images = []
    descriptions = []
    for photo in photos.entries:
        url = photo.urls.regular
        try:
            response = requests.get(url, stream=True)
            image = Image.open(response.raw)
            images.append(image)
            descriptions.append(photo.alt_description)
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
    # print(data.shape)
    # data = data.reshape(-1, 1)
    st.write(len(images))
    st.write(len(descriptions))
    embedding = umap.UMAP().fit_transform(data)
    clusters = clustering_algorithm.fit_predict(embedding)
    return clusters

# Streamlit app structure
st.title("Product Image Clustering with Unsplash")

query = st.text_input("Enter search query for Unsplash:")

if st.button("Gather Images"):
    images, descriptions = gather_images(query)

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
