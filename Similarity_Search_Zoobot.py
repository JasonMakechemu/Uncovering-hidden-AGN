#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:00:45 2024

@author: jason
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image
import math

import requests
from io import BytesIO
from sklearn.metrics.pairwise import manhattan_distances


# Load feature vectors table
feature_vectors_table = pd.read_csv('/mmfs1/storage/users/makechem/representations of COSMOS-Web.csv')

# Load image locations table
#change to the table of xray sources that fit visual criteria 
image_locations_table  = pd.read_csv('/mmfs1/storage/users/makechem/Unique_COSMOS-Web_Sources (32pix).csv')    


# Assuming feature_vectors_table is a DataFrame where each row is a feature vector
# Calculate pairwise cosine similarity
similarity_matrix = cosine_similarity(feature_vectors_table, feature_vectors_table)



# Example: Retrieve top 5 similar images for the first image
query_image_index = 7674  # Change this index as needed
num_similar_images = 60
num_columns = 10  # Number of columns in the grid
num_rows = math.ceil((num_similar_images + 1) / num_columns)



# Get similarity scores for the query image
similarity_scores = similarity_matrix[query_image_index]

# Get indices of top similar images (excluding the query image itself)
similar_image_indices = np.argsort(similarity_scores)[::-1][1:num_similar_images + 1]

# Retrieve the paths/URLs of similar images from image_locations_table
similar_image_paths = image_locations_table.iloc[similar_image_indices]['file_loc'].tolist()





# Retrieve path of the query image
query_image_path = image_locations_table.iloc[query_image_index]['file_loc']
query_image = Image.open(query_image_path)

# Display query image and similar images
fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))

# Flatten axes array for easy indexing
axes = axes.flatten()

# Plot query image
axes[0].imshow(query_image)
axes[0].set_title('Query Image')
axes[0].axis('off')

print(similar_image_paths)

# Plot similar images
for i, image_path in enumerate(similar_image_paths):
    image = Image.open(similar_image_paths[i])
    axes[i + 1].imshow(image)
    axes[i + 1].set_title(f'Similar Image {i + 1}')
    axes[i + 1].axis('off')

# Hide any unused subplots
for j in range(len(similar_image_paths) + 1, len(axes)):
    axes[j].axis('off')


# Adjust spacing to make the fit tight between images
plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust wspace and hspace as needed
plt.savefig('grid of similar AGN.png', dpi=600)
plt.tight_layout()
plt.show()




# Display the indices and their corresponding similarity scores from the features table (optional)
for idx in similar_image_indices:
    print(f"Image Index: {idx}, Similarity Score: {similarity_scores[idx]}")






#%%






