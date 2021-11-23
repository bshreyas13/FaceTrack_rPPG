# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 00:44:51 2021

@author: local_test
"""
import os
import numpy as np
import tensorflow as tf
from modules.videodatasethandler import VideoDatasetHandler


## Function gets subset of data , splits data to obtain train , val, test sets ##
## Return: 3 lists of video_folder names sXX_trialXX ##
def getSets( motion_path, subset=0.01 , val_split=0.1, test_split=0.2):
    vdh = VideoDatasetHandler()
    all_videos = os.listdir(motion_path)
    in_data = vdh.getSubset(all_videos,subset)    
    train_set, val_set, test_set = vdh.splitData(in_data,val_split, test_split)  
    return train_set, val_set, test_set 

def getDatasets(model, appearance_path,motion_path, labels_path, x_shape, y_shape,batch_size =50, timesteps = 5 , img_size = (300,215,3),subset=0.01,val_split=0.1,test_split=0.2):
    
    train_set, val_set, test_set = getSets(motion_path,subset,val_split,test_split)
    vdh = VideoDatasetHandler()
    
    types=((tf.float64, tf.float64), tf.float64)
    shapes = ({x_shape, x_shape}, y_shape)
    ## Train, Val, Test Dataset for Appearance Stream
    datagen_train = vdh.dataGenerator(model, train_set, appearance_path , motion_path, labels_path, batch_size =50, timesteps = 5 , img_size = (300,215,3))
    train_ds = tf.data.Dataset.from_generator(
        generator = datagen_train, 
        output_types= types, 
        output_shapes = shapes)
    
    datagen_val = vdh.dataGenerator(model, val_set, appearance_path, motion_path, labels_path, batch_size =50, timesteps = 5 , img_size = (300,215,3))
    val_ds = tf.data.Dataset.from_generator(
        generator = datagen_val, 
        output_types=types, 
        output_shapes = shapes)

    datagen_test= vdh.dataGenerator(model, test_set, appearance_path, motion_path, labels_path, batch_size =50, timesteps = 5 , img_size = (300,215,3))
    test_ds= tf.data.Dataset.from_generator(
        generator = datagen_test, 
        output_types=types, 
        output_shapes = shapes)
    
    return train_ds, val_ds, test_ds

def addNormalizationLayer(ds):
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    normalized_ds = ds.map(lambda x, y: (normalization_layer(x), y))
    
    return normalized_ds
        
    
