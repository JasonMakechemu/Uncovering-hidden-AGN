#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:41:25 2024

@author: jason
"""

import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_context('notebook')
from sklearn.metrics import DistanceMetric
from PIL import Image
import matplotlib.ticker as ticker

from sklearn.decomposition import IncrementalPCA


from sklearn.neighbors import NearestNeighbors
#from zoobot import label_metadata
from galaxy_datasets.shared import label_metadata 
import requests
from io import BytesIO



def find_neighbours(X, query_index, n_neighbors=36, metric='manhattan'):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric=metric).fit(X)
    _, indices = nbrs.kneighbors(X[query_index].reshape(1, -1))
    return indices


def show_galaxies(df, n_galaxies=36, label=None):
    if n_galaxies == 8:
        fig, all_axes = plt.subplots(1, 8, figsize=(20, 3.5))
    elif n_galaxies == 11:
        fig, all_axes = plt.subplots(1, 11, figsize=(20, 4.5))
    elif n_galaxies == 12:
        fig, axes = plt.subplots(2, 6, figsize=(20, 7))
        all_axes = [ax for row in axes for ax in row]
    else:
        fig, axes = plt.subplots(6, 6, figsize=(20, 20))
        all_axes = [ax for row in axes for ax in row]
    for ax_n, ax in enumerate(all_axes):
        
        response = requests.get(df.iloc[ax_n]['file_loc'].replace('beta1', 'beta'))
        im = Image.open(BytesIO(response.content))        
        
        #im = Image.open(df.iloc[ax_n]['file_loc'].replace('beta1', 'beta'))
        
        crop_pixels = 120
        initial_size = 424 # assumed, careful
        (left, upper, right, lower) = (crop_pixels, crop_pixels, initial_size-crop_pixels, initial_size-crop_pixels)
        im = im.crop((left, upper, right, lower))

        ax.imshow(np.array(im))
        
        if ax_n == 0:
            ax.patch.set_edgecolor('green')  
            ax.patch.set_linewidth('14')  
            # can't just disable axis as also disables border, do manually instead
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_minor_locator(ticker.NullLocator())
            if label:
                ax.set_ylabel(label, labelpad=7, fontsize=14)
        else:
            ax.axis('off')
            
    fig.tight_layout(pad=1.)
    
    return fig


df = pd.read_parquet('/mmfs1/storage/users/makechem/representations of CEERS.parquet')

print(df['file_loc'])
#%%

feature_cols = [col for col in df if col.startswith('feat')]

print(feature_cols)


def get_embed(features, n_components, save=''):
    embedder = IncrementalPCA(n_components=n_components)
    embed = embedder.fit_transform(features) 
     # no train/test needed as unsupervised
    if len(save) > 0:
        plt.plot(embedder.explained_variance_)  # 5 would probably do?
        plt.savefig(save)
        plt.close()
    return embed


features = df[feature_cols].values

print(features)




#%%




vote_df = pd.read_parquet('/mmfs1/storage/users/makechem/jwst-ceers-v0-5-aggregated-class-with-compound-fracs_test_copy.parquet')

fraction_cols = [col for col in vote_df.columns if 'frac' in col.lower()]


vote_df = vote_df[vote_df['q00_smooth_or_featured_total_count_JWST'] > 34]



nans_by_col = np.mean(pd.isna(vote_df[fraction_cols]).values, axis=0)
for col, me in zip(fraction_cols, nans_by_col):
    print(col, '{:.3f}'.format(me))



fraction_cols_not_eob = [col for col in fraction_cols if 'disk_edgeon' not in col]


has_any_nan = np.any(pd.isna(vote_df[fraction_cols_not_eob]).values, axis=1)
has_any_nan.mean()  # from 60% to 40% (almost all from not spiral)

vote_features = vote_df.loc[~has_any_nan][fraction_cols_not_eob].values
vote_features.shape


vote_features_id = vote_df.loc[~has_any_nan]['id_JWST'].reset_index(drop=True)
vote_features_id





vote_embed = get_embed(vote_features, n_components=10)

has_retired_nonnan_votes = vote_df['id_JWST'].isin(vote_features_id)
df_with_votes = vote_df[has_retired_nonnan_votes].reset_index(drop=True)
embed_with_votes = embed[has_retired_nonnan_votes]  # will also be ordered the same as was vote_df and df were merged together (i.e. sorted by iauname)



def get_most_similar_galaxies(df, embed, galaxy_index):
    assert galaxy_index # != 0
    indices = find_neighbours(embed, galaxy_index, metric='euclidean')
    return df.iloc[np.squeeze(indices)]



galaxy_index = 100


most_similar_rep_galaxies = get_most_similar_galaxies(df_with_votes, embed_with_votes, galaxy_index)
fig = show_galaxies(most_similar_rep_galaxies, n_galaxies=n_galaxies, label=tag_label)  # first is itself


plt.show()
plt.close()













