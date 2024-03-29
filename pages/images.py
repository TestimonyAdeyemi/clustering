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

# def fetch_images(query, count=10):
#     url = f"https://api.unsplash.com/search/photos/?query={query}&client_id={UNSPLASH_ACCESS_KEY}&per_page={count}"
#     response = requests.get(url)
#     data = response.json()
#     return [photo['urls']['regular'] for photo in data['results']], [photo.get('description') for photo in data['results']]

def fetch_images(query, count=10):
    #per_page = max(count, 30)
    per_page = 30  # Fetch at least 30 images to ensure we have enough
    url = f"https://api.unsplash.com/search/photos/?query={query}&client_id={UNSPLASH_ACCESS_KEY}&per_page={per_page}"
    response = requests.get(url)
    data = response.json()
    results = data['results'][:count]  # Return only the first count images
    return [photo['urls']['regular'] for photo in results], [photo.get('description') for photo in results]


import urllib.parse

# def fetch_images(query, count=10):
#     encoded_query = urllib.parse.quote(query)
#     url = f"https://api.unsplash.com/search/photos/?query={encoded_query}&client_id={UNSPLASH_ACCESS_KEY}&per_page={count}"
#     response = requests.get(url)
#     data = response.json()
#     return [photo['urls']['regular'] for photo in data['results']], [photo.get('description') for photo in data['results']]



def download_images(urls, folder_name='images'):
    os.makedirs(folder_name, exist_ok=True)
    for i, url in enumerate(urls):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save(f"{folder_name}/image_{i}.jpg")

def cluster_images(image_folder, descriptions, num_clusters=3, use_text_clustering=False, resize_dims=(100, 100)):
    image_files = os.listdir(image_folder)
    valid_descriptions = [desc for desc in descriptions if desc is not None]
    images = []
    for image_file, desc in zip(image_files, descriptions):
        if desc is None:
            continue
        img = Image.open(os.path.join(image_folder, image_file))
        img = img.resize(resize_dims)
        img_array = np.array(img)
        images.append(img_array.flatten())
    
    if use_text_clustering:
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=2)
        text_features = vectorizer.fit_transform(valid_descriptions)
        km = KMeans(n_clusters=num_clusters)
        km.fit(text_features)
        return [file for file, desc in zip(image_files, descriptions) if desc is not None], None, km.labels_
    else:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(images)
        return [file for file, desc in zip(image_files, descriptions) if desc is not None], kmeans.labels_, None

# Streamlit UI
st.title("Image Clustering from Unsplash")

query = st.text_input("Enter search query:", "jewelery")
#num_images = st.slider("Number of images to fetch:", 1, 20, 10)
num_images = 40
num_clusters = st.slider("Number of clusters:", 2, 10, 3)
use_text_clustering = st.checkbox("Cluster by Text Description")

if st.button("Fetch and Cluster Images"):
    #st.write("Fetching images...")
    image_urls, descriptions = fetch_images(query, num_images)
    #st.write(f"Fetched images.")
    #st.write(f"Fetched {len(image_urls)} images.")


# if st.button("Fetch and Cluster Images"):
#     st.write("Fetching images...")
#     image_urls, descriptions = fetch_images(query, 20)  # Adjust num_images to 20
#     st.write(f"Fetched {len(image_urls)} images.")

    
    st.write("Downloading images...")
    download_images(image_urls)
    #st.write("Images downloaded successfully.")
    
    # st.write("Clustering images...")
    # image_files, image_labels, text_labels = cluster_images('images', descriptions, num_clusters, use_text_clustering)
    # st.write("Images clustered successfully.")

    st.write("Clustering images...")
    image_files, image_labels, text_labels = cluster_images('images'[:20], descriptions[:20], num_clusters, use_text_clustering)
    #st.write("Images clustered successfully.")

    
    if image_labels is not None:
        st.write("Cluster labels (Image Content):")
        st.write(image_labels)
    if text_labels is not None:
        st.write("Cluster labels (Text Description):")
        st.write(text_labels)
    
    # st.write("Displaying images with labels:")
    # if image_labels is not None:
    #     for image_file, image_label in zip(image_files, image_labels):
    #         st.image(f'images/{image_file}', caption=f'Image Cluster: {image_label}', use_column_width=True)
    # elif text_labels is not None:
    #     for image_file, text_label in zip(image_files, text_labels):
    #         st.image(f'images/{image_file}', caption=f'Text Cluster: {text_label}', use_column_width=True)
    # else:
    #     for image_file in image_files:
    #         st.image(f'images/{image_file}', use_column_width=True)

    # st.write("Displaying images with labels and descriptions:")
    # if image_labels is not None:
    #     for image_file, image_label, description in zip(image_files, image_labels, descriptions):
    #         st.image(f'images/{image_file}', caption=f'Image Cluster: {image_label}', use_column_width=True)
    #         if description:
    #             st.write(description)
    #         else:
    #             st.write("No description available.")
    #         st.write("---")
    # elif text_labels is not None:
    #     for image_file, text_label, description in zip(image_files, text_labels, descriptions):
    #         st.image(f'images/{image_file}', caption=f'Text Cluster: {text_label}', use_column_width=True)
    #         if description:
    #             st.write(description)
    #         else:
    #             st.write("No description available.")
    #         st.write("---")
    # else:
    #     for image_file, description in zip(image_files, descriptions):
    #         st.image(f'images/{image_file}', use_column_width=True)
    #         if description:
    #             st.write(description)
    #         else:
    #             st.write("No description available.")
    #         st.write("---")
        
    st.write("Displaying images with labels and descriptions:")
    for image_file, image_label, description in zip(image_files, image_labels, descriptions):
        st.image(f'images/{image_file}', caption=f'Image Cluster: {image_label}', use_column_width=True)
        if description:
            st.write(description)
        else:
            st.write("No description available.")
        st.write("---")

        
