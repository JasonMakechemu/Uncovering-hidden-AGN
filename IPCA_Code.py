#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:40:15 2024

@author: husmak
"""

import os
import glob
from umap import UMAP

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook')
import pandas as pd
import numpy as np
from PIL import Image

import requests
from io import BytesIO

import umap
from sklearn.decomposition import IncrementalPCA



#%%



def show_thumbnails(galaxy_images, galaxy_data_umap, num_bins):
    
    # get galaxy_data_umap bin edges with hist2d
    _, hx, hy = np.histogram2d(galaxy_data_umap[:, 0], galaxy_data_umap[:, 1], bins=[num_bins, num_bins])
    
    # find bin index for all galaxies in galaxy_data_umap
    hx_indices = np.digitize(galaxy_data_umap[:, 0], hx)
    hy_indices = np.digitize(galaxy_data_umap[:, 1], hy)
    h_indices = np.stack([hx_indices, hy_indices], axis=1)

    # place every image in df into a dict like {(x_index, y_index): loc} according to its bin index
    # only the latest will be kept (one per bin)
    images_map = {}
    for image_loc, indices in zip(galaxy_images, h_indices):
        images_map[tuple(indices)] = image_loc

    fig, axes = plt.subplots(ncols=num_bins+1, nrows=num_bins+1, figsize=(50, 50))

    for row in axes:
        for ax in row:
            ax.axis('off')

    for indices, image_loc in images_map.items():
        ax = axes[indices[0]-1, indices[1]-1]  # indices from digitize are 1-indexed
        
        
        #image = Image.open(image_loc) #for locally stored images

        response = requests.get(image_loc)
        image = Image.open(BytesIO(response.content))


        ax.imshow(np.array(image))
        ax.axis('off')
        
    #fig.tight_layout(pad=0.0)
    fig.savefig("CEERS Visualisation.png", dpi=600)    
    
    return images_map
#%%
   


# FOR COSMOS - Load your galaxy data (assuming it's a numpy array with shape (n_galaxies, 640))

'''
galaxy_data = pd.read_csv('/mmfs1/storage/users/makechem/representations of COSMOS-Web.csv') # where to get representations
print(galaxy_data)


unique_sources = pd.read_csv("/mmfs1/storage/users/makechem/Unique_COSMOS-Web_Sources (32pix).csv") # where to get images


import re


number_in_image = []


for i in range(len(galaxy_data)):
    text_string = unique_sources['file_loc'][i]  # Replace with your thumbnail file format and path        
    text_string = text_string.strip()  # Handle whitespace (optional)

    # Find the first occurrence of a number using regular expressions
    match = re.findall(r"\d+", text_string)

    if len(match) >= 2:
      # Extract the number and convert it to an integer
      number = match[1]
      number_in_image.append(number)  # Output: 123
    else:
      print("No number found in the string")



galaxy_images = []

for i in range(len(galaxy_data)):
    #thumbnail_path = unique_sources['file_loc'][i]
    
    thumbnail_path = f"/mmfs1/storage/users/makechem/Unique COSMOS-Web Sources/Unique COSMOS-Web Source " + str(number_in_image[i]) + ".png"  # Replace with your thumbnail file format and path        
      
    image = Image.open(thumbnail_path)
    galaxy_images.append(thumbnail_path)

galaxy_images = np.array(galaxy_images)

'''






# Load your galaxy data (assuming it's a numpy array with shape (n_galaxies, 640))


galaxy_data = pd.read_csv('/mmfs1/storage/users/makechem/representations of CEERS.csv') # where to get representations
print(galaxy_data)


unique_sources = pd.read_csv("/mmfs1/storage/users/makechem/jwst-ceers-v0-5-aggregated-class-with-compound-fracs_test.csv") # where to get images


galaxy_images = []

for i in range(len(galaxy_data)):
    thumbnail_path = unique_sources['file_loc'][i]
    
    #thumbnail_path = f"/mmfs1/storage/users/makechem/Unique COSMOS-Web Sources/Unique COSMOS-Web Source " + str(number_in_image[i]) + ".png"  # Replace with your thumbnail file format and path        
      
    response = requests.get(thumbnail_path)
    image = Image.open(BytesIO(response.content))
    galaxy_images.append(thumbnail_path)

galaxy_images = np.array(galaxy_images)



  
# Step 2: Incremental PCA
ipca = IncrementalPCA(n_components=15)
galaxy_data_ipca = ipca.fit_transform(galaxy_data)
  
# Step 3: UMAP
umap = UMAP(n_components=2)
galaxy_data_umap = umap.fit_transform(galaxy_data_ipca)
  
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(galaxy_data_umap[:, 1], -galaxy_data_umap[:, 0], alpha=.3, s=1.)
ax.axis('off')
fig.savefig('CEERS UMAP Visualisation.png', dpi=600)




show_thumbnails(galaxy_images, galaxy_data_umap, 12)









    