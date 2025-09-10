#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:58:34 2024

@author: jason
"""

import os
import requests
import numpy as np
import webbrowser
import pandas as pd
import seaborn as sns
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw


sns.set_context('notebook')


'''
open the predictions made on the CEERS test images and converts the csv to parquet,
then ensures that the id_str is an int.
'''

preds = pd.read_csv('/Users/jason/Desktop/Predictions_on_CEERS_Test_Data.csv') # actually on validation set
preds.to_parquet('predictions_on_CEERS_Test_Data.parquet')

preds["id_str"] = preds["id_str"].astype(int) 


'''
opens the ceers test .csv which contains the image location and the volunterr votes

opens the ceers train .csv which contains the image location and the volunterr votes

concatenates the train and test .csv files into a dataframe called 'df'

and add the CEERS predictions to the df too

df contains CEERS prediction, and the CEERS volunteer classifications

'''
test = pd.read_csv('/Users/jason/Desktop/jwst-ceers-v0-5-aggregated-class-with-compound-fracs_test.csv')
test.to_parquet('jwst-ceers-v0-5-aggregated-class-with-compound-fracs_test.parquet')

train = pd.read_csv('/Users/jason/Desktop/jwst-ceers-v0-5-aggregated-class-with-compound-fracs_training.csv')
train.to_parquet('jwst-ceers-v0-5-aggregated-class-with-compound-fracs_training.parquet')

validate = pd.read_csv('/Users/jason/Desktop/jwst-ceers-v0-5-aggregated-class-with-compound-fracs_validation.csv')
validate.to_parquet('jwst-ceers-v0-5-aggregated-class-with-compound-fracs_validation.parquet')

df = pd.concat([train, test, validate]) 

df = pd.merge(df, preds, on='id_str', how='left', validate='one_to_one') #merges the train and test files with the predictions file.




'''
cosmos-web predictions
'''


cosmos_preds = pd.read_csv('/Users/jason/Desktop/Predictions_on_COSMOS-Web_Data.csv')
cosmos_preds.to_parquet('Predictions_on_COSMOS-Web_Data.parquet')

cosmos_preds["id_str"] = cosmos_preds["id_str"].astype(int)



#%%

#make sum of counts for smooth, featured, artifacts for volunteers and zoobot, then take difference of the sums.
#make fractional difference plot between smooth, featured, artefact because they should total 40 (sum of all classifications)

'''
Plotting fraction of galaxies that are 1. Featured or disk, 2. smooth, 3.Artifacts.

total counts of the answers to the CEERS predicted questions

'''


df['q00_smooth_or_features_total_count_pred'] = df['q00_smooth_or_featured_features_or_disk_count_JWST_pred'] + df['q00_smooth_or_featured_smooth_count_JWST_pred'] + df['q00_smooth_or_featured_artifact_count_JWST_pred']
df['q02_disk_edgeon_total_count_pred'] = df['q02_disk_edgeon_yes_count_JWST_pred'] + df['q02_disk_edgeon_no_count_JWST_pred']
df['q11_rare_features_total_count_pred'] = df['q11_rare_features_dust_lane_count_JWST_pred'] + df['q11_rare_features_irregular_count_JWST_pred'] +df['q11_rare_features_lens_or_arc_count_JWST_pred'] +df['q11_rare_features_nothing_unusual_count_JWST_pred'] +df['q11_rare_features_overlapping_count_JWST_pred'] + df['q11_rare_features_something_else_count_JWST_pred'] +  df['q11_rare_features_ring_count_JWST_pred'] + df['q11_rare_features_diffraction_spikes_count_JWST_pred']   

#fractions of diffraction spike over the total rare features
df['fractions_of_diffraction_spikes'] = (df['q11_rare_features_diffraction_spikes_count_JWST']/df['q11_rare_features_total_count_JWST'])

             
#total counts for COSMOS predicted questions
#cosmos_preds['q11_rare_features_total_count_pred'] = cosmos_preds['q11_rare_features_dust_lane_count_JWST_pred'] + cosmos_preds['q11_rare_features_irregular_count_JWST_pred'] + cosmos_preds['q11_rare_features_lens_or_arc_count_JWST_pred'] + cosmos_preds['q11_rare_features_nothing_unusual_count_JWST_pred'] + cosmos_preds['q11_rare_features_overlapping_count_JWST_pred'] + cosmos_preds['q11_rare_features_something_else_count_JWST_pred'] + cosmos_preds['q11_rare_features_ring_count_JWST_pred'] + cosmos_preds['q11_rare_features_diffraction_spikes_count_JWST_pred']   



#%%
'''
To get fractions for CEERS GZ Volunteer questions.
'''

preds['q00_smooth_or_featured_smooth_frac_JWST_pred'] = preds['q00_smooth_or_featured_smooth_count_JWST_pred'] / (preds['q00_smooth_or_featured_smooth_count_JWST_pred'] + preds['q00_smooth_or_featured_features_or_disk_count_JWST_pred'] + preds['q00_smooth_or_featured_artifact_count_JWST_pred'])
preds['q00_smooth_or_featured_features_or_disk_frac_JWST_pred'] = preds['q00_smooth_or_featured_features_or_disk_count_JWST_pred'] / (preds['q00_smooth_or_featured_smooth_count_JWST_pred'] + preds['q00_smooth_or_featured_features_or_disk_count_JWST_pred'] + preds['q00_smooth_or_featured_artifact_count_JWST_pred'])
preds['q00_smooth_or_featured_artifact_frac_JWST_pred'] = preds['q00_smooth_or_featured_artifact_count_JWST_pred'] / (preds['q00_smooth_or_featured_smooth_count_JWST_pred'] + preds['q00_smooth_or_featured_features_or_disk_count_JWST_pred'] + preds['q00_smooth_or_featured_artifact_count_JWST_pred'])



preds['q01_how_rounded_is_it_cigarshaped_frac_JWST_pred'] = preds['q01_how_rounded_is_it_cigar_count_JWST_pred'] / (preds['q01_how_rounded_is_it_cigar_count_JWST_pred'] + preds['q01_how_rounded_is_it_in_between_count_JWST_pred'] + preds['q01_how_rounded_is_it_completely_round_count_JWST_pred'])
preds['q01_how_rounded_is_it_in_between_frac_JWST_pred'] = preds['q01_how_rounded_is_it_in_between_count_JWST_pred'] / (preds['q01_how_rounded_is_it_cigar_count_JWST_pred'] + preds['q01_how_rounded_is_it_in_between_count_JWST_pred'] + preds['q01_how_rounded_is_it_completely_round_count_JWST_pred'])
preds['q01_how_rounded_is_it_completely_round_frac_JWST_pred'] = preds['q01_how_rounded_is_it_completely_round_count_JWST_pred'] / (preds['q01_how_rounded_is_it_cigar_count_JWST_pred'] + preds['q01_how_rounded_is_it_in_between_count_JWST_pred'] + preds['q01_how_rounded_is_it_completely_round_count_JWST_pred'])


preds['q02_disk_edgeon_yes_frac_JWST_pred'] = preds['q02_disk_edgeon_yes_count_JWST_pred'] / (preds['q02_disk_edgeon_yes_count_JWST_pred'] + preds['q02_disk_edgeon_no_count_JWST_pred'])
preds['q02_disk_edgeon_no_frac_JWST_pred'] = preds['q02_disk_edgeon_no_count_JWST_pred'] / (preds['q02_disk_edgeon_yes_count_JWST_pred'] + preds['q02_disk_edgeon_no_count_JWST_pred'])


preds['q03_bulge_shape_boxy_frac_JWST_pred'] = preds['q03_bulge_shape_boxy_count_JWST_pred'] / (preds['q03_bulge_shape_boxy_count_JWST_pred'] + preds['q03_bulge_shape_round_count_JWST_pred'] + preds['q03_bulge_shape_none_count_JWST_pred'])
preds['q03_bulge_shape_round_frac_JWST_pred'] = preds['q03_bulge_shape_round_count_JWST_pred'] / (preds['q03_bulge_shape_boxy_count_JWST_pred'] + preds['q03_bulge_shape_round_count_JWST_pred'] + preds['q03_bulge_shape_none_count_JWST_pred'])
preds['q03_bulge_shape_none_frac_JWST_pred'] = preds['q03_bulge_shape_none_count_JWST_pred'] / (preds['q03_bulge_shape_boxy_count_JWST_pred'] + preds['q03_bulge_shape_round_count_JWST_pred'] + preds['q03_bulge_shape_none_count_JWST_pred'])

preds['q04_bright_clumps_yes_frac_JWST_pred'] = preds['q04_bright_clumps_yes_count_JWST_pred'] / (preds['q04_bright_clumps_yes_count_JWST_pred'] + preds['q04_bright_clumps_no_count_JWST_pred'])
preds['q04_bright_clumps_no_frac_JWST_pred'] = preds['q04_bright_clumps_no_count_JWST_pred'] / (preds['q04_bright_clumps_yes_count_JWST_pred'] + preds['q04_bright_clumps_no_count_JWST_pred'])


preds['q06_isbar_strong_frac_JWST_pred'] = preds['q06_isbar_strong_count_JWST_pred'] / (preds['q06_isbar_strong_count_JWST_pred'] + preds['q06_isbar_weak_count_JWST_pred'] + preds['q06_isbar_none_count_JWST_pred'])
preds['q06_isbar_weak_frac_JWST_pred'] = preds['q06_isbar_weak_count_JWST_pred'] / (preds['q06_isbar_strong_count_JWST_pred'] + preds['q06_isbar_weak_count_JWST_pred'] + preds['q06_isbar_none_count_JWST_pred'])
preds['q06_isbar_none_frac_JWST_pred'] = preds['q06_isbar_none_count_JWST_pred'] / (preds['q06_isbar_strong_count_JWST_pred'] + preds['q06_isbar_weak_count_JWST_pred'] + preds['q06_isbar_none_count_JWST_pred'])


preds['q07_is_spiral_yes_frac_JWST_pred'] = preds['q07_is_spiral_yes_count_JWST_pred'] / (preds['q07_is_spiral_no_count_JWST_pred'] + 
                                                                                          preds['q07_is_spiral_no_count_JWST_pred'])

preds['q07_is_spiral_no_frac_JWST_pred'] = preds['q07_is_spiral_no_count_JWST_pred'] / (preds['q07_is_spiral_no_count_JWST_pred'] + 
                                                                                        preds['q07_is_spiral_no_count_JWST_pred'])


preds['q08_how_tightly_wound_loose_frac_JWST_pred'] = preds['q08_how_tightly_wound_loose_count_JWST_pred'] / (preds['q08_how_tightly_wound_loose_count_JWST_pred'] + preds['q08_how_tightly_wound_medium_count_JWST_pred'] + preds['q08_how_tightly_wound_tight_count_JWST_pred'])
preds['q08_how_tightly_wound_medium_frac_JWST_pred'] = preds['q08_how_tightly_wound_medium_count_JWST_pred'] / (preds['q08_how_tightly_wound_loose_count_JWST_pred'] + preds['q08_how_tightly_wound_medium_count_JWST_pred'] + preds['q08_how_tightly_wound_tight_count_JWST_pred'])
preds['q08_how_tightly_wound_tight_frac_JWST_pred'] = preds['q08_how_tightly_wound_tight_count_JWST_pred'] / (preds['q08_how_tightly_wound_loose_count_JWST_pred'] + preds['q08_how_tightly_wound_medium_count_JWST_pred'] + preds['q08_how_tightly_wound_tight_count_JWST_pred'])


preds['q09_how_many_spiral_arms_1_frac_JWST_pred'] = preds['q09_how_many_spiral_arms_1_count_JWST_pred'] / (preds['q09_how_many_spiral_arms_1_count_JWST_pred'] + preds['q09_how_many_spiral_arms_2_count_JWST_pred'] + 
                                                                                                            preds['q09_how_many_spiral_arms_3_count_JWST_pred'] + preds['q09_how_many_spiral_arms_4_count_JWST_pred'] +
                                                                                                            preds['q09_how_many_spiral_arms_more_than_4_count_JWST_pred'] + 
                                                                                                            preds['q09_how_many_spiral_arms_cant_tell_count_JWST_pred'])

preds['q09_how_many_spiral_arms_2_frac_JWST_pred'] = preds['q09_how_many_spiral_arms_2_count_JWST_pred'] / (preds['q09_how_many_spiral_arms_1_count_JWST_pred'] + preds['q09_how_many_spiral_arms_2_count_JWST_pred'] + 
                                                                                                            preds['q09_how_many_spiral_arms_3_count_JWST_pred'] + preds['q09_how_many_spiral_arms_4_count_JWST_pred'] +
                                                                                                            preds['q09_how_many_spiral_arms_more_than_4_count_JWST_pred'] + 
                                                                                                            preds['q09_how_many_spiral_arms_cant_tell_count_JWST_pred'])

preds['q09_how_many_spiral_arms_3_frac_JWST_pred'] = preds['q09_how_many_spiral_arms_3_count_JWST_pred'] / (preds['q09_how_many_spiral_arms_1_count_JWST_pred'] + preds['q09_how_many_spiral_arms_2_count_JWST_pred'] + 
                                                                                                            preds['q09_how_many_spiral_arms_3_count_JWST_pred'] + preds['q09_how_many_spiral_arms_4_count_JWST_pred'] +
                                                                                                            preds['q09_how_many_spiral_arms_more_than_4_count_JWST_pred'] + 
                                                                                                            preds['q09_how_many_spiral_arms_cant_tell_count_JWST_pred'])

preds['q09_how_many_spiral_arms_4_frac_JWST_pred'] = preds['q09_how_many_spiral_arms_4_count_JWST_pred'] / (preds['q09_how_many_spiral_arms_1_count_JWST_pred'] + preds['q09_how_many_spiral_arms_2_count_JWST_pred'] + 
                                                                                                            preds['q09_how_many_spiral_arms_3_count_JWST_pred'] + preds['q09_how_many_spiral_arms_4_count_JWST_pred'] +
                                                                                                            preds['q09_how_many_spiral_arms_more_than_4_count_JWST_pred'] + 
                                                                                                            preds['q09_how_many_spiral_arms_cant_tell_count_JWST_pred'])

preds['q09_how_many_spiral_arms_more_than_4_frac_JWST_pred'] = preds['q09_how_many_spiral_arms_more_than_4_count_JWST_pred'] / (preds['q09_how_many_spiral_arms_1_count_JWST_pred'] + preds['q09_how_many_spiral_arms_2_count_JWST_pred'] + 
                                                                                                                                preds['q09_how_many_spiral_arms_3_count_JWST_pred'] + preds['q09_how_many_spiral_arms_4_count_JWST_pred'] +
                                                                                                                                preds['q09_how_many_spiral_arms_more_than_4_count_JWST_pred'] + 
                                                                                                                                preds['q09_how_many_spiral_arms_cant_tell_count_JWST_pred'])

preds['q09_how_many_spiral_arms_cant_tell_frac_JWST_pred'] = preds['q09_how_many_spiral_arms_cant_tell_count_JWST_pred'] / (preds['q09_how_many_spiral_arms_1_count_JWST_pred'] + preds['q09_how_many_spiral_arms_2_count_JWST_pred'] + 
                                                                                                                             preds['q09_how_many_spiral_arms_3_count_JWST_pred'] + preds['q09_how_many_spiral_arms_4_count_JWST_pred'] +
                                                                                                                             preds['q09_how_many_spiral_arms_more_than_4_count_JWST_pred'] + 
                                                                                                                             preds['q09_how_many_spiral_arms_cant_tell_count_JWST_pred'])


preds['q10_is_bulge_no_bulge_frac_JWST_pred'] = preds['q10_is_bulge_no_bulge_count_JWST_pred'] / (preds['q10_is_bulge_no_bulge_count_JWST_pred'] + preds['q10_is_bulge_small_count_JWST_pred'] +
                                                                                                  preds['q10_is_bulge_moderate_count_JWST_pred'] + preds['q10_is_bulge_large_count_JWST_pred'] +
                                                                                                  preds['q10_is_bulge_dominant_count_JWST_pred'])

preds['q10_is_bulge_small_frac_JWST_pred'] = preds['q10_is_bulge_small_count_JWST_pred'] / (preds['q10_is_bulge_no_bulge_count_JWST_pred'] + preds['q10_is_bulge_small_count_JWST_pred'] +
                                                                                                  preds['q10_is_bulge_moderate_count_JWST_pred'] + preds['q10_is_bulge_large_count_JWST_pred'] +
                                                                                                  preds['q10_is_bulge_dominant_count_JWST_pred'])

preds['q10_is_bulge_moderate_frac_JWST_pred'] = preds['q10_is_bulge_moderate_count_JWST_pred'] / (preds['q10_is_bulge_no_bulge_count_JWST_pred'] + preds['q10_is_bulge_small_count_JWST_pred'] +
                                                                                                  preds['q10_is_bulge_moderate_count_JWST_pred'] + preds['q10_is_bulge_large_count_JWST_pred'] +
                                                                                                  preds['q10_is_bulge_dominant_count_JWST_pred'])

preds['q10_is_bulge_large_frac_JWST_pred']  = preds['q10_is_bulge_large_count_JWST_pred'] / (preds['q10_is_bulge_no_bulge_count_JWST_pred'] + preds['q10_is_bulge_small_count_JWST_pred'] +
                                                                                                  preds['q10_is_bulge_moderate_count_JWST_pred'] + preds['q10_is_bulge_large_count_JWST_pred'] +
                                                                                                  preds['q10_is_bulge_dominant_count_JWST_pred'])

preds['q10_is_bulge_dominant_frac_JWST_pred'] = preds['q10_is_bulge_dominant_count_JWST_pred'] / (preds['q10_is_bulge_no_bulge_count_JWST_pred'] + preds['q10_is_bulge_small_count_JWST_pred'] +
                                                                                                  preds['q10_is_bulge_moderate_count_JWST_pred'] + preds['q10_is_bulge_large_count_JWST_pred'] +
                                                                                                  preds['q10_is_bulge_dominant_count_JWST_pred'])


preds['q11_rare_features_nothing_unusual_frac_JWST_pred'] = preds['q11_rare_features_diffraction_spikes_count_JWST_pred'] / (preds['q11_rare_features_dust_lane_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_irregular_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_lens_or_arc_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_nothing_unusual_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_overlapping_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_something_else_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_ring_count_JWST_pred'] + 
                                                                                                                               preds['q11_rare_features_diffraction_spikes_count_JWST_pred'])

preds['q11_rare_features_dust_lane_frac_JWST_pred']= preds['q11_rare_features_dust_lane_count_JWST_pred'] / (preds['q11_rare_features_dust_lane_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_irregular_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_lens_or_arc_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_nothing_unusual_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_overlapping_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_something_else_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_ring_count_JWST_pred'] + 
                                                                                                                               preds['q11_rare_features_diffraction_spikes_count_JWST_pred'])

preds['q11_rare_features_irregular_frac_JWST_pred']= preds['q11_rare_features_irregular_count_JWST_pred'] / (preds['q11_rare_features_dust_lane_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_irregular_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_lens_or_arc_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_nothing_unusual_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_overlapping_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_something_else_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_ring_count_JWST_pred'] + 
                                                                                                                               preds['q11_rare_features_diffraction_spikes_count_JWST_pred'])

preds['q11_rare_features_lens_or_arc_frac_JWST_pred'] = preds['q11_rare_features_lens_or_arc_count_JWST_pred'] / (preds['q11_rare_features_dust_lane_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_irregular_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_lens_or_arc_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_nothing_unusual_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_overlapping_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_something_else_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_ring_count_JWST_pred'] + 
                                                                                                                               preds['q11_rare_features_diffraction_spikes_count_JWST_pred'])

preds['q11_rare_features_overlapping_frac_JWST_pred'] = preds['q11_rare_features_overlapping_count_JWST_pred'] / (preds['q11_rare_features_dust_lane_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_irregular_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_lens_or_arc_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_nothing_unusual_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_overlapping_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_something_else_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_ring_count_JWST_pred'] + 
                                                                                                                               preds['q11_rare_features_diffraction_spikes_count_JWST_pred'])

preds['q11_rare_features_something_else_frac_JWST_pred'] = preds['q11_rare_features_something_else_count_JWST_pred'] / (preds['q11_rare_features_dust_lane_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_irregular_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_lens_or_arc_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_nothing_unusual_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_overlapping_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_something_else_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_ring_count_JWST_pred'] + 
                                                                                                                               preds['q11_rare_features_diffraction_spikes_count_JWST_pred'])

preds['q11_rare_features_ring_frac_JWST_pred'] = preds['q11_rare_features_ring_count_JWST_pred'] / (preds['q11_rare_features_dust_lane_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_irregular_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_lens_or_arc_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_nothing_unusual_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_overlapping_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_something_else_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_ring_count_JWST_pred'] + 
                                                                                                                               preds['q11_rare_features_diffraction_spikes_count_JWST_pred'])

preds['q11_rare_features_diffraction_spikes_frac_JWST_pred'] = preds['q11_rare_features_diffraction_spikes_count_JWST_pred'] / (preds['q11_rare_features_dust_lane_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_irregular_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_lens_or_arc_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_nothing_unusual_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_overlapping_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_something_else_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_ring_count_JWST_pred'] + 
                                                                                                                               preds['q11_rare_features_diffraction_spikes_count_JWST_pred'])



preds['q05_merging_merging_frac_JWST_pred'] = preds['q05_merging_merging_count_JWST_pred'] / (preds['q05_merging_merging_count_JWST_pred'] + preds['q05_merging_major_disturbance_count_JWST_pred'] +
                                                                                             preds['q05_merging_minor_disturbance_count_JWST_pred'] + preds['q05_merging_none_count_JWST_pred'])

preds['q05_merging_major_disturbance_frac_JWST_pred'] = preds['q05_merging_major_disturbance_count_JWST_pred'] / (preds['q05_merging_merging_count_JWST_pred'] + preds['q05_merging_major_disturbance_count_JWST_pred'] +
                                                                                             preds['q05_merging_minor_disturbance_count_JWST_pred'] + preds['q05_merging_none_count_JWST_pred'])

preds['q05_merging_minor_disturbance_frac_JWST_pred'] = preds['q05_merging_minor_disturbance_count_JWST_pred'] / (preds['q05_merging_merging_count_JWST_pred'] + preds['q05_merging_major_disturbance_count_JWST_pred'] +
                                                                                             preds['q05_merging_minor_disturbance_count_JWST_pred'] + preds['q05_merging_none_count_JWST_pred'])

preds['q05_merging_none_frac_JWST_pred'] = preds['q05_merging_none_count_JWST_pred'] / (preds['q05_merging_merging_count_JWST_pred'] + preds['q05_merging_major_disturbance_count_JWST_pred'] +
                                                                                             preds['q05_merging_minor_disturbance_count_JWST_pred'] + preds['q05_merging_none_count_JWST_pred'])




preds['q12_problem_nonstar_artifact_frac_JWST_pred'] =  preds['q12_problem_nonstar_artifact_count_JWST_pred'] / (preds['q12_problem_nonstar_artifact_count_JWST_pred'] + preds['q12_problem_bad_zoom_count_JWST_pred']
                                                                                                                + preds['q12_problem_star_count_JWST_pred'])
preds['q12_problem_bad_zoom_frac_JWST_pred'] = preds['q12_problem_bad_zoom_count_JWST_pred'] / (preds['q12_problem_nonstar_artifact_count_JWST_pred'] + preds['q12_problem_bad_zoom_count_JWST_pred']
                                                                                               + preds['q12_problem_star_count_JWST_pred'])
preds['q12_problem_star_frac_JWST_pred'] = preds['q12_problem_star_count_JWST_pred'] / (preds['q12_problem_nonstar_artifact_count_JWST_pred'] + preds['q12_problem_bad_zoom_count_JWST_pred']
                                                                                       + preds['q12_problem_star_count_JWST_pred'])





#%%
'''
making compound fractions for Zoobot CEERS predictions, e.g. ff_spiral = f_featured * f_notedgeon * f_spiral
'''


newcols = []


# these first ones aren't strictly necessary as they aren't compound, but they're here for completeness
preds['ff_smooth_pred']   = preds['q00_smooth_or_featured_smooth_frac_JWST_pred']
preds['ff_featured_pred'] = preds['q00_smooth_or_featured_features_or_disk_frac_JWST_pred']
preds['ff_artifact_pred'] = preds['q00_smooth_or_featured_artifact_frac_JWST_pred']
newcols.extend(['ff_smooth_pred', 'ff_featured_pred', 'ff_artifact_pred'])

preds['ff_rounded_cigar_pred']        = preds['ff_smooth_pred'] * preds['q01_how_rounded_is_it_cigarshaped_frac_JWST_pred']
preds['ff_rounded_inbetween_pred']    = preds['ff_smooth_pred'] * preds['q01_how_rounded_is_it_in_between_frac_JWST_pred']
preds['ff_rounded_completely_pred']   = preds['ff_smooth_pred'] * preds['q01_how_rounded_is_it_completely_round_frac_JWST_pred']
newcols.extend(['ff_rounded_cigar_pred', 'ff_rounded_inbetween_pred', 'ff_rounded_completely_pred'])

# I don't want to do f_not = 1 - f because that isn't quite right -- f_notedgeon is the compound fraction that it's featured *and* not edge on
# this does mean these compound fractions might not add to 1 within the same question
preds['ff_edgeon_pred']               = preds['ff_featured_pred'] * preds['q02_disk_edgeon_yes_frac_JWST_pred']
preds['ff_notedgeon_pred']            = preds['ff_featured_pred'] * preds['q02_disk_edgeon_no_frac_JWST_pred']
newcols.extend(['ff_edgeon_pred', 'ff_notedgeon_pred'])

# ff_edgeon has ff_featured in it, and so on
preds['ff_edgeonbulge_boxy_pred']     = preds['ff_edgeon_pred'] * preds['q03_bulge_shape_boxy_frac_JWST_pred']
preds['ff_edgeonbulge_round_pred']    = preds['ff_edgeon_pred'] * preds['q03_bulge_shape_round_frac_JWST_pred']
preds['ff_edgeonbulge_none_pred']     = preds['ff_edgeon_pred'] * preds['q03_bulge_shape_none_frac_JWST_pred']
newcols.extend(['ff_edgeonbulge_boxy_pred', 'ff_edgeonbulge_round_pred', 'ff_edgeonbulge_none_pred'])

preds['ff_clumpy_pred']               = (1. - preds['ff_artifact_pred']) * preds['q04_bright_clumps_yes_frac_JWST_pred']
preds['ff_notclumpy_pred']            = (1. - preds['ff_artifact_pred']) * preds['q04_bright_clumps_no_frac_JWST_pred']
newcols.extend(['ff_clumpy_pred', 'ff_notclumpy_pred'])

preds['ff_bar_strong_pred']           = preds['ff_notedgeon_pred'] * preds['q06_isbar_strong_frac_JWST_pred']
preds['ff_bar_weak_pred']             = preds['ff_notedgeon_pred'] * preds['q06_isbar_weak_frac_JWST_pred']
preds['ff_bar_none_pred']             = preds['ff_notedgeon_pred'] * preds['q06_isbar_none_frac_JWST_pred']
newcols.extend(['ff_bar_strong_pred', 'ff_bar_weak_pred', 'ff_bar_none_pred'])

preds['ff_spiral_pred']               = preds['ff_notedgeon_pred'] * preds['q07_is_spiral_yes_frac_JWST_pred']
preds['ff_notspiral_pred']            = preds['ff_notedgeon_pred'] * preds['q07_is_spiral_no_frac_JWST_pred']
newcols.extend(['ff_spiral_pred', 'ff_notspiral_pred'])

preds['ff_spiralwind_loose_pred']         = preds['ff_spiral_pred'] * preds['q08_how_tightly_wound_loose_frac_JWST_pred']
preds['ff_spiralwind_medium_pred']        = preds['ff_spiral_pred'] * preds['q08_how_tightly_wound_medium_frac_JWST_pred']
preds['ff_spiralwind_tight_pred']         = preds['ff_spiral_pred'] * preds['q08_how_tightly_wound_tight_frac_JWST_pred']
preds['wind_param_pred'] = preds['ff_spiralwind_tight_pred'] + (2./3. * preds['ff_spiralwind_medium_pred']) + (1./3. * preds['ff_spiralwind_loose_pred'])
newcols.extend(['ff_spiralwind_loose_pred', 'ff_spiralwind_medium_pred', 'ff_spiralwind_tight_pred', 'wind_param_pred'])

preds['ff_spiralct_1_pred']           = preds['ff_spiral_pred'] * preds['q09_how_many_spiral_arms_1_frac_JWST_pred']
preds['ff_spiralct_2_pred']           = preds['ff_spiral_pred'] * preds['q09_how_many_spiral_arms_2_frac_JWST_pred']
preds['ff_spiralct_3_pred']           = preds['ff_spiral_pred'] * preds['q09_how_many_spiral_arms_3_frac_JWST_pred']
preds['ff_spiralct_4_pred']           = preds['ff_spiral_pred'] * preds['q09_how_many_spiral_arms_4_frac_JWST_pred']
preds['ff_spiralct_more_than_4_pred'] = preds['ff_spiral_pred'] * preds['q09_how_many_spiral_arms_more_than_4_frac_JWST_pred']
preds['ff_spiralct_cant_tell_pred']   = preds['ff_spiral_pred'] * preds['q09_how_many_spiral_arms_cant_tell_frac_JWST_pred']
newcols.extend(['ff_spiralct_1_pred', 'ff_spiralct_2_pred', 'ff_spiralct_3_pred', 'ff_spiralct_4_pred', 'ff_spiralct_more_than_4_pred', 'ff_spiralct_cant_tell_pred'])

preds['ff_bulgestr_nobulge_pred']     = preds['ff_notedgeon_pred'] * preds['q10_is_bulge_no_bulge_frac_JWST_pred']
preds['ff_bulgestr_small_pred']       = preds['ff_notedgeon_pred'] * preds['q10_is_bulge_small_frac_JWST_pred']
preds['ff_bulgestr_moderate_pred']    = preds['ff_notedgeon_pred'] * preds['q10_is_bulge_moderate_frac_JWST_pred']
preds['ff_bulgestr_large_pred']       = preds['ff_notedgeon_pred'] * preds['q10_is_bulge_large_frac_JWST_pred']
preds['ff_bulgestr_dominant_pred']    = preds['ff_notedgeon_pred'] * preds['q10_is_bulge_dominant_frac_JWST_pred']
# create a single "bulge parameter"
preds['B_param_pred'] = preds['ff_bulgestr_dominant_pred'] + (0.75 * preds['ff_bulgestr_large_pred']) + (0.5 * preds['ff_bulgestr_moderate_pred']) + (0.25 * preds['ff_bulgestr_small_pred'])
newcols.extend(['ff_bulgestr_nobulge_pred', 'ff_bulgestr_small_pred', 'ff_bulgestr_moderate_pred', 'ff_bulgestr_large_pred', 'ff_bulgestr_dominant_pred', 'B_param_pred'])


preds['ff_rarefeat_none_pred']        = (1. - preds['ff_artifact_pred']) * preds['q11_rare_features_nothing_unusual_frac_JWST_pred']
preds['ff_rarefeat_dustlane_pred']    = (1. - preds['ff_artifact_pred']) * preds['q11_rare_features_dust_lane_frac_JWST_pred']
preds['ff_rarefeat_irregular_pred']   = (1. - preds['ff_artifact_pred']) * preds['q11_rare_features_irregular_frac_JWST_pred']
preds['ff_rarefeat_lens_or_arc_pred'] = (1. - preds['ff_artifact_pred']) * preds['q11_rare_features_lens_or_arc_frac_JWST_pred']
preds['ff_rarefeat_overlap_pred']     = (1. - preds['ff_artifact_pred']) * preds['q11_rare_features_overlapping_frac_JWST_pred']
preds['ff_rarefeat_other_pred']       = (1. - preds['ff_artifact_pred']) * preds['q11_rare_features_something_else_frac_JWST_pred']
preds['ff_rarefeat_ring_pred']        = (1. - preds['ff_artifact_pred']) * preds['q11_rare_features_ring_frac_JWST_pred']
preds['ff_rarefeat_diffspikes_pred']  = (1. - preds['ff_artifact_pred']) * preds['q11_rare_features_diffraction_spikes_frac_JWST_pred']
newcols.extend(['ff_rarefeat_none_pred', 'ff_rarefeat_dustlane_pred', 'ff_rarefeat_irregular_pred', 'ff_rarefeat_lens_or_arc_pred', 'ff_rarefeat_overlap_pred', 'ff_rarefeat_other_pred', 'ff_rarefeat_ring_pred', 'ff_rarefeat_diffspikes_pred'])

preds['ff_merging_pred']              = (1. - preds['ff_artifact_pred']) * preds['q05_merging_merging_frac_JWST_pred']
preds['ff_majordisturb_pred']         = (1. - preds['ff_artifact_pred']) * preds['q05_merging_major_disturbance_frac_JWST_pred']
preds['ff_minordisturb_pred']         = (1. - preds['ff_artifact_pred']) * preds['q05_merging_minor_disturbance_frac_JWST_pred']
preds['ff_notmerging_pred']           = (1. - preds['ff_artifact_pred']) * preds['q05_merging_none_frac_JWST_pred']
newcols.extend(['ff_merging', 'ff_majordisturb', 'ff_minordisturb', 'ff_notmerging'])

preds['ff_nonstar_artifact_pred']     = preds['ff_artifact_pred'] * preds['q12_problem_nonstar_artifact_frac_JWST_pred']
preds['ff_bad_zoom_pred']             = preds['ff_artifact_pred'] * preds['q12_problem_bad_zoom_frac_JWST_pred']
preds['ff_star_pred']                 = preds['ff_artifact_pred'] * preds['q12_problem_star_frac_JWST_pred']
newcols.extend(['ff_nonstar_artifact_pred', 'ff_bad_zoom_pred', 'ff_star_pred'])


'''

outfile_agg = '/Users/jason/Desktop/Predictions_on_CEERS_Test_Data.csv' # actually on validation set


outfile_agg_withff = outfile_agg.replace(".csv", "-with-compound-fracs.csv")

new_labels_ordered_compound = new_labels_ordered.copy()
new_labels_ordered_compound.extend(newcols)

preds[new_labels_ordered_compound].to_csv(outfile_agg_withff)
print("Aggregated classification file with new column labels etc plus compound vote fractions saved to %s ." % outfile_agg_withff)
'''







'''
CEERS smooth fraction delta 
CEERS Features or Disk fraction delta 
CEERS Artifact fraction delta 
'''


plt.figure()

plt.hist((df['q00_smooth_or_featured_smooth_count_JWST_pred']/df['q00_smooth_or_features_total_count_pred']) - (df['q00_smooth_or_featured_smooth_count_JWST']/df['q00_smooth_or_featured_total_count_JWST']), bins=20, color='red')
plt.xlabel('"CEERS Smooth" fraction delta')
plt.ylabel('Galaxies')
plt.savefig("smooth fraction delta.png", dpi=600)

plt.figure()

plt.hist((df['q00_smooth_or_featured_features_or_disk_count_JWST_pred']/df['q00_smooth_or_features_total_count_pred']) - (df['q00_smooth_or_featured_features_or_disk_count_JWST']/df['q00_smooth_or_featured_total_count_JWST']), bins=20, color='blue')
plt.xlabel('"Features or Disk" fraction delta')
plt.ylabel('Galaxies')


plt.figure()

plt.hist((df['q00_smooth_or_featured_artifact_count_JWST_pred']/df['q00_smooth_or_features_total_count_pred']) - (df['q00_smooth_or_featured_artifact_count_JWST']/df['q00_smooth_or_featured_total_count_JWST']), bins=20, color='green')
plt.xlabel('"Artifact" fraction delta')
plt.ylabel('Galaxies')
plt.savefig("artifact fraction delta.png", dpi=600)



'''
CEERS Disk edge-on galaxies
'''

plt.figure()

plt.hist((df['q02_disk_edgeon_yes_count_JWST_pred']/df['q02_disk_edgeon_total_count_pred']) - (df['q02_disk_edgeon_yes_count_JWST']/df['q02_disk_edgeon_total_count_JWST']), bins=20, color='black')
plt.xlabel('"Disk edge-on" fraction delta')
plt.ylabel('Galaxies')
plt.savefig("edge on disk fraction delta.png", dpi=600)


#%%

'''
CEERS diffraction spikes delta.

Difference between the fractions of diffraction spikes to the total rare features from the
Zoobot CEERS predictions.
'''

plt.figure()

plt.hist((df['q11_rare_features_diffraction_spikes_count_JWST_pred']/df['q11_rare_features_total_count_pred']) - (df['q11_rare_features_diffraction_spikes_count_JWST']/df['q11_rare_features_total_count_JWST']), bins=20, color='black')
plt.xlabel('"CEERS Diffraction spikes" delta')
plt.ylabel('Galaxies')
plt.savefig("diffraction spike delta.png", dpi=600)



'''
CEERS diffraction spikes as fractions of total rare features. Zoobot predictioons,
followed by GZ volunteers.

Lastly, histogram of compound fractions of diffraction spikes.
'''

plt.figure()

plt.hist((df['q11_rare_features_diffraction_spikes_count_JWST_pred']/df['q11_rare_features_total_count_pred']), bins=20, color='purple')
plt.xlabel('"CEERS Diffraction spikes" predicted fraction')
plt.ylabel('Galaxies')


'''
featured fraction count. Zoobot CEERS pred. 
'''
plt.figure()

plt.hist((preds['ff_rarefeat_diffspikes_pred']), bins=20, color='purple')
plt.xlabel('"CEERS Diffraction spikes" predicted feat. fraction')
plt.ylabel('Galaxies')



'''
CEERS diffraction spikes delta.

Difference between the fractions of diffraction spikes to the total rare features from the
CEERS volunteers.
'''

plt.figure()

plt.hist((df['q11_rare_features_diffraction_spikes_count_JWST']/df['q11_rare_features_total_count_JWST']), bins=20, color='red')
plt.xlabel('"CEERS Diffraction spikes" volunteer fraction')
plt.ylabel('Galaxies')


'''
featured fraction count. CEERS volunteers 
'''
plt.figure()

plt.hist(df['ff_rarefeat_diffspikes'], bins=20, color='red')
plt.xlabel('"CEERS Diffraction spikes" feat. fraction')
plt.ylabel('Galaxies')









#%%


'''
PREDICTED FEATURED FRACTIONS OF DIFFRACTION SPIKES AGAINST PREDICTED PROBABILITY
'''

plt.figure()

plt.scatter(preds['ff_rarefeat_diffspikes_pred'], preds['q11_rare_features_diffraction_spikes_count_JWST_pred'] / (preds['q11_rare_features_dust_lane_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_irregular_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_lens_or_arc_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_nothing_unusual_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_overlapping_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_something_else_count_JWST_pred'] +
                                                                                                                               preds['q11_rare_features_ring_count_JWST_pred'] + 
                                                                                                                               preds['q11_rare_features_diffraction_spikes_count_JWST_pred']))

plt.ylabel('CEERS Pred Diff. spikes / total rare feat. Zoobot')
plt.xlabel('CEERS Pred Diff. spikes compound frac')
plt.savefig('CEERS frac vs prob Pred.png', dpi=600)



#%%

#sum of all votes of rare objects
df['q11_rare_features_total_votes_count_JWST'] = df['q11_rare_features_dust_lane_count_JWST'] + df['q11_rare_features_irregular_count_JWST'] +df['q11_rare_features_lens_or_arc_count_JWST'] +df['q11_rare_features_nothing_unusual_count_JWST'] +df['q11_rare_features_overlapping_count_JWST'] + df['q11_rare_features_something_else_count_JWST'] +  df['q11_rare_features_ring_count_JWST'] + df['q11_rare_features_diffraction_spikes_count_JWST']  





'''
similarity search
umap for AGN

Scatter plot of diffraction spikes as a fraction of rare feature for the CEERS volunteer classifications.
chen, ho, li 2024
Kocevski 2024 LRDs
Greene 2023, Labbe 2023 LRDs

PSFEx to make synthetic psf models.
'''

links3 = np.array(df['file_loc'])

fig, ax = plt.subplots()
sc3 = ax.scatter((df['q11_rare_features_diffraction_spikes_count_JWST']/df['q11_rare_features_total_votes_count_JWST']), (df['q11_rare_features_diffraction_spikes_count_JWST_pred']/df['q11_rare_features_total_count_pred']))

# Function to open link on click
def onpick(event):
    ind = event.ind[0]
    webbrowser.open(links3[ind])
    
# Connect the click event with the onpick function
fig.canvas.mpl_connect('pick_event', onpick)

# Make the scatter plot pickable
sc3.set_picker(True)

# Show plot
plt.show()

plt.xlabel('CEERS Diff. spikes / total rare feat.')
plt.ylabel('Diff. spikes / total rare feat. Zoobot')
plt.savefig('CEERS frac vs prob.png', dpi=600)










'''
Scatter plot of diffraction spikes featured fraction against fraction of rare feature for the Zoobot predictions.
'''

links2 = np.array(df['file_loc'])

fig, ax = plt.subplots()
sc2 = ax.scatter(df['ff_rarefeat_diffspikes'], (df['q11_rare_features_diffraction_spikes_count_JWST_pred']/df['q11_rare_features_total_count_pred']))

# Function to open link on click
def onpick(event):
    ind = event.ind[0]
    webbrowser.open(links2[ind])

# Connect the click event with the onpick function
fig.canvas.mpl_connect('pick_event', onpick)

# Make the scatter plot pickable
sc2.set_picker(True)

# Show plot
plt.show()

plt.xlabel('ff_rarefeat_diffspikes')
plt.ylabel('Zoobot Diff. spikes / total rare feat. Zoobot pred')
plt.savefig('CEERS frac vs prob.png', dpi=600)


















'''
Plot of diffraction spikes as compound fractions from CEERS volunteers against
the fraction of diffraction spikes against rare features for the Zoobot predictions.
'''





links1 = np.array(df['file_loc'])

fig, ax = plt.subplots()
sc1 = ax.scatter(df['ff_rarefeat_diffspikes'], (df['q11_rare_features_diffraction_spikes_count_JWST']/df['q11_rare_features_total_votes_count_JWST']))

# Function to open link on click
def onpick(event):
    ind = event.ind[0]
    webbrowser.open(links1[ind])


# Connect the click event with the onpick function
fig.canvas.mpl_connect('pick_event', onpick)

# Make the scatter plot pickable
sc1.set_picker(True)

# Show plot
plt.show()


plt.xlabel('ff_rarefeat_diffspikes')
plt.ylabel('CEERS Diff. spikes / total rare feat.')
plt.savefig('CEERS frac vs prob.png', dpi=600)
















'''
html target for opening window python

make compund fractions for ceers and zoobot, and plot against each other for 
every question.

similarity search umap
'''



links = np.array(df['file_loc'][5377:6144])


#on validation set only
fig, ax = plt.subplots()
sc = ax.scatter(preds['ff_rarefeat_diffspikes_pred'], df['ff_rarefeat_diffspikes'][5377:6144])

# Function to open link on click
def onpick(event):
    ind = event.ind[0]
    webbrowser.open(links[ind])


# Connect the click event with the onpick function
fig.canvas.mpl_connect('pick_event', onpick)

# Make the scatter plot pickable
sc.set_picker(True)

# Show plot
plt.show()


plt.xlabel('ff_rarefeat_diffspikes_pred')
plt.ylabel('ff_rarefeat_diffspikes')
plt.savefig('CEERS frac vs prob Pred.png', dpi=600)


#%%

'''
#make histogram of sum of GZ rare vote fraction for each rare feature, not compound fraction

Is it that there are more rare features overall, or, it is just not diffraction spikes for the right hand
side objects.

Is the diffraction spike probability low, or the rare features are high?

How often does zoobot say something is rare vs. how often the volunteers say that?

df['q11_rare_features_total_count_JWST'] - not the sum of all votes of rare objects, it's the sum of the number of people who voted. 
Manually add votes of rare objects across all sources to get accurate representation

Activation actions - DONE


'''

#%%

'''
use feat frac for ceers and preds for the histograms

'''

#diffraction spikes over total rare features count!

plt.figure()
plt.hist(df['q11_rare_features_diffraction_spikes_count_JWST']/df['q11_rare_features_total_votes_count_JWST'], bins=20)
plt.xlabel('CEERS diff. spikes / total rare feat.')
plt.ylabel('Count')

plt.figure()
plt.hist(df['q11_rare_features_dust_lane_frac_JWST']/df['q11_rare_features_total_votes_count_JWST'], bins=20)
plt.xlabel('CEERS dust lane / total rare feat.')
plt.ylabel('Count')



plt.figure()
plt.hist(df['q11_rare_features_nothing_unusual_frac_JWST']/df['q11_rare_features_total_votes_count_JWST'], bins=20)
plt.xlabel('CEERS nothing unusual / total rare feat.')
plt.ylabel('Count')


plt.figure()
plt.hist(df['q11_rare_features_irregular_frac_JWST']/df['q11_rare_features_total_votes_count_JWST'], bins=20)
plt.xlabel('CEERS irregular / total rare feat.')
plt.ylabel('Count')



plt.figure()
plt.hist(df['q11_rare_features_lens_or_arc_frac_JWST']/df['q11_rare_features_total_votes_count_JWST'])
plt.xlabel('CEERS lens or arc / total rare feat.')
plt.ylabel('Count')







plt.figure()
plt.hist(df['q11_rare_features_diffraction_spikes_count_JWST_pred']/df['q11_rare_features_total_count_pred'])
plt.xlabel('Zoobot diff. spikes / total rare feat.')
plt.ylabel('Count')

'''
plt.figure()
plt.hist(df['q11_rare_features_dust_lane_frac_JWST_pred']/df['q11_rare_features_total_count_pred'])
plt.xlabel('CEERS dust lane / total rare feat.')
plt.ylabel('Count')


plt.figure()
plt.hist(df['q11_rare_features_nothing_unusual_frac_JWST_pred']/df['q11_rare_features_total_count_pred'])
plt.xlabel('CEERS nothing unusual / total rare feat.')
plt.ylabel('Count')


plt.figure()
plt.hist(df['q11_rare_features_irregular_frac_JWST_pred']/df['q11_rare_features_total_count_pred'])
plt.xlabel('CEERS irregular / total rare feat.')
plt.ylabel('Count')



plt.figure()
plt.hist(df['q11_rare_features_lens_or_arc_frac_JWST_pred']/df['q11_rare_features_total_count_pred'])
plt.xlabel('CEERS lens or arc / total rare feat.')
plt.ylabel('Count')

plt.figure()
plt.hist(df['q11_rare_features_overlapping_frac_JWST_pred']/df['q11_rare_features_total_count_pred'])
plt.xlabel('CEERS overlapping / total rare feat.')
plt.ylabel('Count')



plt.figure()
plt.hist(df['q11_rare_features_something_else_frac_JWST_pred']/df['q11_rare_features_total_count_pred'])
plt.xlabel('CEERS smth. else / total rare feat.')
plt.ylabel('Count')

plt.figure()
plt.hist(df['q11_rare_features_ring_frac_JWST_pred']/df['q11_rare_features_total_count_pred'])
plt.xlabel('CEERS ring frac. / total rare feat.')
plt.ylabel('Count')




plt.figure()
plt.hist( (df['q11_rare_features_ring_frac_JWST_pred'] + df['q11_rare_features_dust_lane_frac_JWST_pred'] +
           df['q11_rare_features_nothing_unusual_frac_JWST_pred'] + df['q11_rare_features_irregular_frac_JWST_pred'] +
           df['q11_rare_features_lens_or_arc_frac_JWST_pred'] + df['q11_rare_features_overlapping_frac_JWST_pred'] + 
           df['q11_rare_features_something_else_frac_JWST_pred'] + df['q11_rare_features_ring_frac_JWST_pred'])  /  df['q11_rare_features_total_count_pred'])
plt.xlabel('CEERS ring frac. / total rare feat.')
plt.ylabel('Count')
'''

#%%






'''

plt.figure()
plt.scatter((df['q11_rare_features_diffraction_spikes_count_JWST_pred']/df['q11_rare_features_total_count_pred']), df['ff_rarefeat_diffspikes'])
plt.xlabel('COSMOS Diff. spikes volunteer frac')
plt.ylabel('COSMOS Diff. spikes probability')
plt.savefig('COSMOS pred frac vs prob.png', dpi=600)
'''


'''
COSMOS-Web diffraction spikes fractions predicted fraction.
'''
'''
plt.figure()

plt.hist((cosmos_preds['q11_rare_features_diffraction_spikes_count_JWST_pred']/cosmos_preds['q11_rare_features_total_count_pred']), bins=20, color='orange') #cosmos pred
plt.xlabel('"COSMOS-Web Diffraction spikes" predicted fraction')
plt.ylabel('Galaxies')
'''


'''
sorts df from highest to lowest based on CEERS volunteer fractions on whether
a certain object is a diffraction spike or not.

get those with a fraction of 0.5 and appends them to an array.

Images are opened then plotted.

'''
#%%

'''
sorted_frac = df.sort_values(by=['ff_rarefeat_diffspikes'], ascending=False)


diff_spike_images = []
diff_spike_ff = []
diff_spike_probab = []


for i in range(len(df)):
    
    if sorted_frac['ff_rarefeat_diffspikes'][i] > 0.0:
        
        diff_spike_images.append(sorted_frac['file_loc'][i])
        diff_spike_ff.append(sorted_frac['ff_rarefeat_diffspikes'][i])
        diff_spike_probab.append(df['q11_rare_features_diffraction_spikes_count_JWST'][i]/df['q11_rare_features_total_count_JWST'][i])
        
       
#%%       
print(str(diff_spike_ff))       
        
       
#%%

for i in range(len(diff_spike_images)):
    response = requests.get(diff_spike_images[i])
    img = Image.open(BytesIO(response.content))
    
    fig, ax = plt.subplots(1)
    ax.imshow(img, origin='lower')
    ax.text(10, 20, f'ceers_volunteer_frac {round(diff_spike_ff[i], 2)}', bbox={'facecolor': 'white', 'pad': 1}, size=8)
    ax.text(10, 450, f'ceers_volunteer_frac {round(diff_spike_probab[i], 2)}', bbox={'facecolor': 'white', 'pad': 1}, size=8)

'''
    
#print(sorted_frac['file_loc'][1])



#%%

'''
how is zoobot featured fraction above 1??
'''


#on validation set only
plt.subplots()
plt.scatter(preds['ff_smooth_pred'], df['ff_smooth'][5377:6144])

plt.xlabel('ff_smooth_pred')
plt.ylabel('ff_smooth')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_featured_pred'], df['ff_featured'][5377:6144])

plt.xlabel('ff_featured_pred')
plt.ylabel('ff_featured')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_artifact_pred'], df['ff_artifact'][5377:6144])

plt.xlabel('ff_artifact_pred')
plt.ylabel('ff_artifact')




#%%




#on validation set only
plt.subplots()
plt.scatter(preds['ff_rounded_cigar_pred'], df['ff_rounded_cigar'][5377:6144])

plt.xlabel('ff_rounded_cigar_pred')
plt.ylabel('ff_rounded_cigar')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_rounded_inbetween_pred'], df['ff_rounded_inbetween'][5377:6144])

plt.xlabel('ff_rounded_inbetween_pred')
plt.ylabel('ff_rounded_inbetween')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_rounded_completely_pred'], df['ff_rounded_completely'][5377:6144])

plt.xlabel('ff_rounded_completely_pred')
plt.ylabel('ff_rounded_completely')



#%%

#on validation set only
plt.subplots()
plt.scatter(preds['ff_edgeon_pred'], df['ff_edgeon'][5377:6144])

plt.xlabel('ff_edgeon_pred')
plt.ylabel('ff_edgeon')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_notedgeon_pred'], df['ff_notedgeon'][5377:6144])


plt.xlabel('ff_notedgeon_pred')
plt.ylabel('ff_notedgeon')


#%%


#on validation set only
plt.subplots()
plt.scatter(preds['ff_edgeonbulge_boxy_pred'], df['ff_edgeonbulge_boxy'][5377:6144])

plt.xlabel('ff_edgeonbulge_boxy')
plt.ylabel('ff_notedgeon')

#on validation set only
plt.subplots()
plt.scatter(preds['ff_edgeonbulge_round_pred'], df['ff_edgeonbulge_round'][5377:6144])

plt.xlabel('ff_edgeonbulge_round_pred')
plt.ylabel('ff_edgeonbulge_round')

#on validation set only
plt.subplots()
plt.scatter(preds['ff_edgeonbulge_none_pred'], df['ff_edgeonbulge_none'][5377:6144])

plt.xlabel('ff_edgeonbulge_none_pred')
plt.ylabel('ff_edgeonbulge_none')




#%%


#on validation set only
plt.subplots()
plt.scatter(preds['ff_clumpy_pred'], df['ff_clumpy'][5377:6144])

plt.xlabel('ff_clumpy_pred')
plt.ylabel('ff_clumpy')



#on validation set only
plt.subplots()
plt.scatter(preds['ff_notclumpy_pred'], df['ff_notclumpy'][5377:6144])

plt.xlabel('ff_notclumpy_pred')
plt.ylabel('ff_notclumpy')



#%%

#on validation set only
plt.subplots()
plt.scatter(preds['ff_bar_strong_pred'], df['ff_bar_strong'][5377:6144])

plt.xlabel('ff_bar_strong_pred')
plt.ylabel('ff_bar_strong')



#on validation set only
plt.subplots()
plt.scatter(preds['ff_bar_weak_pred'], df['ff_bar_weak'][5377:6144])

plt.xlabel('ff_bar_weak_pred')
plt.ylabel('ff_bar_weak')



#on validation set only
plt.subplots()
plt.scatter(preds['ff_bar_none_pred'], df['ff_bar_none'][5377:6144])

plt.xlabel('ff_bar_none_pred')
plt.ylabel('ff_bar_none')



#%%



#on validation set only
plt.subplots()
plt.scatter(preds['ff_spiral_pred'], df['ff_spiral'][5377:6144])

plt.xlabel('ff_spiral_pred')
plt.ylabel('ff_spiral')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_notspiral_pred'], df['ff_notspiral'][5377:6144])

plt.xlabel('ff_notspiral_pred')
plt.ylabel('ff_notspiral')



#%%

#on validation set only
plt.subplots()
plt.scatter(preds['ff_spiralwind_loose_pred'], df['ff_spiralwind_loose'][5377:6144])

plt.xlabel('ff_spiralwind_loose_pred')
plt.ylabel('ff_spiralwind_loose')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_spiralwind_medium_pred'], df['ff_spiralwind_medium'][5377:6144])

plt.xlabel('ff_spiralwind_medium_pred')
plt.ylabel('ff_spiralwind_medium')

#on validation set only
plt.subplots()
plt.scatter(preds['ff_spiralwind_tight_pred'], df['ff_spiralwind_tight'][5377:6144])

plt.xlabel('ff_spiralwind_tight_pred')
plt.ylabel('ff_spiralwind_tight')


#%%

#on validation set only
plt.subplots()
plt.scatter(preds['ff_spiralct_1_pred'], df['ff_spiralct_1'][5377:6144])

plt.xlabel('ff_spiralct_1_pred')
plt.ylabel('ff_spiralct_1')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_spiralct_2_pred'], df['ff_spiralct_2'][5377:6144])

plt.xlabel('ff_spiralct_2_pred')
plt.ylabel('ff_spiralct_2')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_spiralct_3_pred'], df['ff_spiralct_3'][5377:6144])

plt.xlabel('ff_spiralct_3_pred')
plt.ylabel('ff_spiralct_3')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_spiralct_4_pred'], df['ff_spiralct_4'][5377:6144])

plt.xlabel('ff_spiralct_4_pred')
plt.ylabel('ff_spiralct_4')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_spiralct_more_than_4_pred'], df['ff_spiralct_more_than_4'][5377:6144])

plt.xlabel('ff_spiralct_more_than_4_pred')
plt.ylabel('ff_spiralct_more_than_4')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_spiralct_cant_tell_pred'], df['ff_spiralct_cant_tell'][5377:6144])

plt.xlabel('ff_spiralct_cant_tell_pred')
plt.ylabel('ff_spiralct_cant_tell')



#%%

#on validation set only
plt.subplots()
plt.scatter(preds['ff_bulgestr_nobulge_pred'], df['ff_bulgestr_nobulge'][5377:6144])

plt.xlabel('ff_bulgestr_nobulge_pred')
plt.ylabel('ff_bulgestr_nobulge')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_bulgestr_small_pred'], df['ff_bulgestr_small'][5377:6144])

plt.xlabel('ff_bulgestr_small_pred')
plt.ylabel('ff_bulgestr_small')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_bulgestr_moderate_pred'], df['ff_bulgestr_moderate'][5377:6144])

plt.xlabel('ff_bulgestr_moderate_pred')
plt.ylabel('ff_bulgestr_moderate')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_bulgestr_large_pred'], df['ff_bulgestr_large'][5377:6144])

plt.xlabel('ff_bulgestr_large_pred')
plt.ylabel('ff_bulgestr_large')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_bulgestr_dominant_pred'], df['ff_bulgestr_dominant'][5377:6144])

plt.xlabel('ff_bulgestr_dominant_pred')
plt.ylabel('ff_bulgestr_dominant')




#%%


'''
Plotting all featured fractions from CEERS against those from Zoobot.
'''


#on validation set only
plt.subplots()
plt.scatter(preds['ff_rarefeat_none_pred'], df['ff_rarefeat_none'][5377:6144])

plt.xlabel('ff_rarefeat_none_pred')
plt.ylabel('ff_rarefeat_none')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_rarefeat_dustlane_pred'], df['ff_rarefeat_dustlane'][5377:6144])

plt.xlabel('ff_rarefeat_dustlane_pred')
plt.ylabel('ff_rarefeat_dustlane')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_rarefeat_irregular_pred'], df['ff_rarefeat_irregular'][5377:6144])

plt.xlabel('ff_rarefeat_irregular_pred')
plt.ylabel('ff_rarefeat_irregular')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_rarefeat_lens_or_arc_pred'], df['ff_rarefeat_lens_or_arc'][5377:6144])

plt.xlabel('ff_rarefeat_lens_or_arc_pred')
plt.ylabel('ff_rarefeat_lens_or_arc')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_rarefeat_overlap_pred'], df['ff_rarefeat_overlap'][5377:6144])

plt.xlabel('ff_rarefeat_overlap_pred')
plt.ylabel('ff_rarefeat_overlap')


#on validation set only
plt.subplots()
plt.scatter(preds['ff_rarefeat_ring_pred'], df['ff_rarefeat_ring'][5377:6144])

plt.xlabel('ff_rarefeat_ring_pred')
plt.ylabel('ff_rarefeat_ring')

#on validation set only
plt.subplots()
plt.scatter(preds['ff_rarefeat_diffspikes_pred'], df['ff_rarefeat_diffspikes'][5377:6144])

plt.xlabel('ff_rarefeat_diffspikes_pred')
plt.ylabel('ff_rarefeat_diffspikes')




#%%

figure, axis = plt.subplots(2, 2, figsize=(5, 4)) 
figure.tight_layout()

#on validation set only
axis[0, 0].scatter(preds['ff_merging_pred'], df['ff_merging'][5377:6144])

axis[0, 0].set_xlabel('ff_merging_pred')
axis[0, 0].set_ylabel('ff_merging')
axis[0, 0].set_title('Featured Fractions Merging')


#on validation set only
axis[0, 1].scatter(preds['ff_majordisturb_pred'], df['ff_majordisturb'][5377:6144])

axis[0, 1].set_xlabel('ff_majordisturb_pred')
axis[0, 1].set_ylabel('ff_majordisturb')
axis[0, 1].set_title('Featured Fractions Major Disturbance')




#on validation set only
axis[1, 0].scatter(preds['ff_minordisturb_pred'], df['ff_minordisturb'][5377:6144])

axis[1, 0].set_xlabel('ff_minordisturb_pred')
axis[1, 0].set_ylabel('ff_minordisturb')
axis[1, 0].set_title('Featured Fractions Minor Disturbance')



#on validation set only
axis[1, 1].scatter(preds['ff_notmerging_pred'], df['ff_notmerging'][5377:6144])

axis[1, 1].set_xlabel('ff_notmerging_pred')
axis[1, 1].set_ylabel('ff_notmerging')
axis[1, 1].set_title('Featured Fractions Not Merging')

figure.savefig('disturbed.png', dpi=600)


#%%

figure, axis = plt.subplots(2, 2, figsize=(5, 4)) 
figure.tight_layout()

#on validation set only
axis[0, 0].scatter(preds['ff_nonstar_artifact_pred'], df['ff_nonstar_artifact'][5377:6144])

axis[0, 0].set_xlabel('ff_nonstar_artifact_pred')
axis[0, 0].set_ylabel('ff_nonstar_artifact')
axis[0, 0].set_title('Featured Fractions Non Star Artifact')


#on validation set only
axis[0, 1].scatter(preds['ff_bad_zoom_pred'], df['ff_bad_zoom'][5377:6144])

axis[0, 1].set_xlabel('ff_bad_zoom_pred')
axis[0, 1].set_ylabel('ff_bad_zoom')
axis[0, 1].set_title('Featured Fractions Bad Zoom')


#on validation set only
axis[1, 0].scatter(preds['ff_star_pred'], df['ff_star'][5377:6144])

axis[1, 0].set_xlabel('ff_star_pred')
axis[1, 0].set_ylabel('ff_star')
axis[1, 0].set_title('Featured Fractions Star')
















