#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 08:20:58 2024

@author: jason
"""

import pandas as pd

csv_files = ['Similar sources COSMOS AGN_3752.csv',
             'Similar sources COSMOS AGN_5032.csv',
             'Similar sources COSMOS AGN_5804.csv',
             'Similar sources COSMOS AGN_6014.csv',
             'Similar sources COSMOS AGN_6121.csv',
             'Similar sources COSMOS AGN_6210.csv',
             'Similar sources COSMOS AGN_6357.csv',
             'Similar sources COSMOS AGN_6403.csv',
             'Similar sources COSMOS AGN_6724.csv',
             'Similar sources COSMOS AGN_6957.csv',
             'Similar sources COSMOS AGN_7016.csv',
             'Similar sources COSMOS AGN_7024.csv',
             'Similar sources COSMOS AGN_7116.csv',
             'Similar sources COSMOS AGN_7186.csv',
             'Similar sources COSMOS AGN_7245.csv',
             'Similar sources COSMOS AGN_7341.csv',
             'Similar sources COSMOS AGN_7444.csv',
             'Similar sources COSMOS AGN_7482.csv',
             'Similar sources COSMOS AGN_7685.csv',
             'Similar sources COSMOS AGN_7811.csv',
             'Similar sources COSMOS AGN_7813.csv',
             'Similar sources COSMOS AGN_7832.csv',
             'Similar sources COSMOS AGN_7836.csv',
             'Similar sources COSMOS AGN_7867.csv',
             'Similar sources COSMOS AGN_7915.csv',
             'Similar sources COSMOS AGN_7927.csv',
             'Similar sources COSMOS AGN_7960.csv',
             'Similar sources COSMOS AGN_7982.csv',
             'Similar sources COSMOS AGN_7995.csv',
             'Similar sources COSMOS AGN_8004.csv',
             'Similar sources COSMOS AGN_8005.csv',
             'Similar sources COSMOS AGN_8020.csv',
             'Similar sources COSMOS AGN_8061.csv',
             'Similar sources COSMOS AGN_8087.csv',
             'Similar sources COSMOS AGN_8102.csv',
             'Similar sources COSMOS AGN_8115.csv',
             'Similar sources COSMOS AGN_8122.csv',
             'Similar sources COSMOS AGN_8123.csv',
             'Similar sources COSMOS AGN_8150.csv',
             'Similar sources COSMOS AGN_8153.csv',
             'Similar sources COSMOS AGN_8155.csv',
             'Similar sources COSMOS AGN_8172.csv',
             'Similar sources COSMOS AGN_8192.csv',
             'Similar sources COSMOS AGN_8208.csv',
             'Similar sources COSMOS AGN_8214.csv',
             'Similar sources COSMOS AGN_8258.csv',
             'Similar sources COSMOS AGN_8283.csv',
             'Similar sources COSMOS AGN_8291.csv'
 ]



dataframes = [pd.read_csv(file) for file in csv_files]

combined_df = pd.concat(dataframes)
combined_df = combined_df.drop_duplicates(subset=['NUMBER'])  # Remove duplicates based on two columns

combined_df.to_csv('Similarity Search No Duplicates.csv', index=False)










