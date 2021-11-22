# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 00:44:51 2021

@author: local_test
"""
import os
import numpy as np
import tensorflow as tf
from modules.videodatasethandler import VideoDatasetHandler


## Then calls datagen to obtain samples ##
## calls train_test_plot to train and evaluate model ##
##  model,in_data, data_path, labels_path,  batch_size =50, time_steps = 5 , img_size = (300,215,3)

## Function gets subset of data , splits data to obtain train , val, test sets ##
## Return: 3 lists of video_folder names sXX_trialXX ##
def getSets( roi_path, subset=0.01 , val_split=0.1, test_split=0.2):
    vdh = VideoDatasetHandler()
    all_videos = os.listdir(roi_path)
    in_data = vdh.getSubset(all_videos,subset)    
    train_set, val_set, test_set = vdh.splitData(in_data,val_split, test_split)  
    return train_set, val_set, test_set 

def getDatasets(model, appearance_path,motion_path, labels_path, train_set, val_set, test_set , x_shape, y_shape):
    vdh = VideoDatasetHandler()
    ## Train, Val, Test Dataset for Appearance Stream
    datagen_train_ap = vdh.dataGenerator(model, train_set, appearance_path ,labels_path)
    train_ds_ap = tf.data.Dataset.from_generator(
        generator=datagen_train_ap, 
        output_types=(np.float64, np.float64), 
        output_shapes=(x_shape, y_shape))
    
    datagen_val_ap = vdh.dataGenerator(model, val_set, appearance_path ,labels_path)
    val_ds_ap = tf.data.Dataset.from_generator(
        generator=datagen_val_ap, 
        output_types=(np.float64, np.float64), 
        output_shapes=(x_shape, y_shape))

    datagen_test_ap = vdh.dataGenerator(model, test_set, appearance_path ,labels_path)
    test_ds_ap = tf.data.Dataset.from_generator(
        generator=datagen_test_ap, 
        output_types=(np.float64, np.float64), 
        output_shapes=(x_shape, y_shape))
    
    ## Train, Val, Test Dataset for motion Stream
    datagen_train_mo = vdh.dataGenerator(model, train_set, motion_path ,labels_path)
    train_ds_mo = tf.data.Dataset.from_generator(
        generator=datagen_train_mo, 
        output_types=(np.float64, np.float64), 
        output_shapes=(x_shape, y_shape))
    
    datagen_val_mo = vdh.dataGenerator(model, val_set, motion_path ,labels_path)
    val_ds_mo = tf.data.Dataset.from_generator(
        generator=datagen_val_mo, 
        output_types=(np.float64, np.float64), 
        output_shapes=(x_shape, y_shape))

    datagen_test_mo = vdh.dataGenerator(model, test_set, motion_path ,labels_path)
    test_ds_mo = tf.data.Dataset.from_generator(
        generator=datagen_test_mo, 
        output_types=(np.float64, np.float64), 
        output_shapes=(x_shape, y_shape))
    
    return train_ds_ap, val_ds_ap, test_ds_ap, train_ds_mo, val_ds_mo, test_ds_mo
    
    
