import streamlit as st
import requests
import os
import shutil
import numpy as np
from PIL import Image
from io import BytesIO
from sklearn.cluster import KMeans
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Define your Unsplash API access key
UNSPLASH_ACCESS_KEY = "MFcvfNHZ-u_kRAhK4oHT2uIIYtWVTxI-pk8qbr4sIR8"

def fetch_images(query, count=10):
    url = f"https://api.unsplash.com/search/photos/?query={query}&client_id={UNSPLASH_ACCESS_KEY}&per_page={count}"
    response = requests.get(url)
    data = response.json()
    return [photo['urls']['regular'] for photo in data['results']], [photo['description'] for photo in data['results']]

def download_images(urls, folder_name='images'):
    os.makedirs(folder_name, exist_ok=True)
    for i, url in enumerate(urls):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save(f"{folder_name}/image_{i}.jpg")

def cluster_images(image_folder, descriptions, num_clusters=3, resize_dims=(100, 100)):
    image_files = os.listdir(image_folder)
    images = []
    for image_file in image_files:
        img = Image.open(os.path.join(image_folder, image_file))
        img = img.resize(resize_dims)
        img_array = np.array(img)
        images.append(img_array.flatten())
    
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(images)
    
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=2)
    text_features = vectorizer.fit_transform(descriptions)
    
    km = KMeans(n_clusters=num_clusters)
    km.fit(text_features)
    
    return image_files, kmeans.labels_, km.labels_

# Streamlit UI
st.title("Image Clustering from Unsplash")

query = st.text_input("Enter search query:", "landscape")
num_images = st.slider("Number of images to fetch:", 1, 20, 10)
num_clusters = st.slider("Number of clusters:", 2, 10, 3)

if st.button("Fetch and Cluster Images"):
    st.write("Fetching images...")
    image_urls, descriptions = fetch_images(query, num_images)
    st.write(f"Fetched {len(image_urls)} images.")
    
    st.write("Downloading images...")
    download_images(image_urls)
    st.write("Images downloaded successfully.")
    
    st.write("Clustering images...")
    image_files, labels, text_labels = cluster_images('images', descriptions, num_clusters)
    st.write("Images clustered successfully.")
    
    st.write("Cluster labels (Image Content):")
    st.write(labels)
    
    st.write("Cluster labels (Text Description):")
    st.write(text_labels)
    
    st.write("Displaying images with labels:")
    for image_file, label, text_label in zip(image_files, labels, text_labels):
        st.image(f'images/{image_file}', caption=f'Content Cluster: {label}, Text Cluster: {text_label}', use_column_width=True)
