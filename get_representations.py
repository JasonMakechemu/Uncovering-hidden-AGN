#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:32:51 2024

@author: husmak
"""

import pandas as pd
import logging
import os
import timm

from zoobot.pytorch.training import finetune, representations
from zoobot.pytorch.estimators import define_model
from zoobot.pytorch.predictions import predict_on_catalog
from zoobot.pytorch.training import finetune
from zoobot.shared import load_predictions, schemas


def main(catalog, save_dir):

    '''
    comment out when getting images from online, uncomment if from a directory.
    assert all([os.path.isfile(x) for x in catalog['file_loc']])
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    '''
    
    finetuned_model = finetune.FinetuneableZoobotTree.load_from_checkpoint('/mmfs1/storage/users/makechem/GZ_Finetuned_Predictions/checkpoints/Epoch 18 Finetuned and Trained on CEERS - COSMOS Preds.ckpt')           
    encoder = finetuned_model.encoder
    model = representations.ZoobotEncoder(encoder=encoder)
    
    encoder_dim = define_model.get_encoder_dim(model.encoder, channels=3)
    label_cols = [f'feat_{n}' for n in range(encoder_dim)]
    save_loc = os.path.join(save_dir, 'predictions on COSMOS-Web.hdf5')

    accelerator = 'gpu'  # or 'gpu' if available
    batch_size = 64
    resize_after_crop = 224

    datamodule_kwargs = {'batch_size': batch_size, 'resize_after_crop': resize_after_crop} #batch size, channels, depth, size, width
    trainer_kwargs = {'devices': 1, 'accelerator': accelerator}
    predict_on_catalog.predict(
        catalog,
        model,
        n_samples=1,
        label_cols=label_cols,
        save_loc=save_loc,
        datamodule_kwargs=datamodule_kwargs,
        trainer_kwargs=trainer_kwargs,
    )

    return save_loc


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)

    # use this demo dataset
    # TODO change this to wherever you'd like, it will auto-download
    #data_dir = '/mmfs1/storage/users/makechem/Unique COSMOS-Web Sources'
    catalog = pd.read_csv('/mmfs1/storage/users/makechem/Unique_COSMOS-Web_Sources (32pix).csv')
    print(catalog.head())
    # zoobot expects id_str and file_loc columns, so add these if needed

    # save the representations here
    # TODO change this to wherever you'd like
    save_dir = os.path.join('/mmfs1/storage/users/makechem/')

    representations_loc = main(catalog, save_dir)
    rep_df = load_predictions.single_forward_pass_hdf5s_to_df(representations_loc)
    rep_df.to_csv('representations of COSMOS-Web.hdf5')  
    print(rep_df)
    
    
    
    
    
    
    
    
    
    