#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:31:26 2024

@author: husmak
"""

import logging
import os
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split

from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule

#from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier
from zoobot.pytorch.training import finetune
from zoobot.pytorch.predictions import predict_on_catalog
from zoobot.shared.schemas import gz_jwst_schema


import timm


encoder = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_nano', pretrained=True, num_classes=0)


"""
Example for finetuning Zoobot on counts of volunteer responses throughout a complex decision tree (here, GZ CANDELS).
Useful if you are running a Galaxy Zoo campaign with many questions and answers.
Probably you are in the GZ collaboration if so!
Also useful if you are running a simple yes/no citizen science project on e.g. the Zooniverse app

See also:
- finetune_binary_classification.py to finetune on class (0 or 1) labels
"""


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    schema = gz_jwst_schema

    # TODO you will want to replace these paths with your own paths
    # I'm being a little lazy and leaving my if/else for local/cluster training here,
    # this is often convenient for debugging
    #if os.path.isdir('/share/nas2'):  # run on cluster
    
    '''
    if os.path.isdir('/mmfs1/storage/users/makechem'):  # run on cluster

        repo_dir = '/share/nas2/walml/repos'
        data_download_dir = '/share/nas2/walml/repos/_data/demo_gz_candels'
        accelerator = 'gpu'
        devices = 1
        batch_size = 64  
        prog_bar = False
        max_galaxies = None
    else:  # test locally
        repo_dir = '/Users/user/repos'
        data_download_dir = '/Users/user/repos/galaxy-datasets/roots/demo_gz_candels'
        accelerator = 'cpu'
        devices = None
        batch_size = 32 # 32 with resize=224, 16 at 380
        prog_bar = True
        # max_galaxies = 256
        max_galaxies = None
    '''

    # pd.DataFrame with columns 'id_str' (unique id), 'file_loc' (path to image),
    # and label_cols (e.g. smooth-or-featured-cd_smooth) with count responses
    #train_and_val_catalog, _ = demo_gz_candels(root=data_download_dir, train=True, download=True)
    
    #test_catalog, _ = demo_gz_candels(root=data_download_dir, train=True, download=True)
    #train_catalog, val_catalog = train_test_split(train_and_val_catalog, test_size=0.3)

    train_catalog  = pd.read_csv('/mmfs1/storage/users/makechem/jwst-ceers-v0-5-aggregated-class-with-compound-fracs_training.csv')
    
    test_catalog  = pd.read_csv('/mmfs1/storage/users/makechem/jwst-ceers-v0-5-aggregated-class-with-compound-fracs_test.csv')
      
    #now testing on COSMOS-Web
    cosmos_web_catalog  = pd.read_csv('/mmfs1/storage/users/makechem/Unique_COSMOS-Web_Sources (32pix).csv')    
    
    val_catalog  = pd.read_csv('/mmfs1/storage/users/makechem/jwst-ceers-v0-5-aggregated-class-with-compound-fracs_validation.csv')
    

    resize_after_crop = 224  # must match how checkpoint below was trained
    #may need to custom transform images
    batch_size = 64
    accelerator = 'gpu'
    
    datamodule = GalaxyDataModule(
        label_cols=schema.label_cols,
        train_catalog=train_catalog,
        val_catalog=val_catalog,
        test_catalog=test_catalog,
        batch_size=batch_size,
        # uses default_augs
        resize_after_crop=resize_after_crop  
    )


    model = finetune.FinetuneableZoobotTree(
            name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano',
            schema=schema,
            n_blocks=5,
            learning_rate=1e-4,
            lr_decay=0.5
        )
    
    
    #save_dir = os.path.join(
        #repo_dir, f'gz-decals-classifiers/results/finetune_{np.random.randint(1e8)}')
    
    save_dir = os.path.join('/mmfs1/storage/users/makechem/', 'GZ_Finetuned_Predictions')
    
    
    # can do logger=None or, to use wandb:
    from pytorch_lightning.loggers import WandbLogger
    logger = WandbLogger(project='GalaxyZoo_JWST_Finetuning', name='full_tree_CEERS_COMSOS_Web_Pred')

    trainer = finetune.get_trainer(save_dir=save_dir, logger=logger, accelerator=accelerator)
    trainer.fit(model, datamodule)

    # now save predictions on test set to evaluate performance
    datamodule_kwargs = {'batch_size': batch_size, 'resize_after_crop': resize_after_crop}
    trainer_kwargs = {'devices': 4, 'accelerator': accelerator}
    predict_on_catalog.predict(
        test_catalog,
        model,
        n_samples=1,
        label_cols=schema.label_cols,
        save_loc=os.path.join(save_dir, 'Predictions_on_CEERS_CHECK_Data.csv'),
        datamodule_kwargs=datamodule_kwargs,
        trainer_kwargs=trainer_kwargs
    )
    
    

    