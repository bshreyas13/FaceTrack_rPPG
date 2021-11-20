# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 23:31:07 2021

@author: bshreyas
"""

import cv2
import pathlib
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf

## This module has utils to handle video datasets ##

class VideoDatasetHandler:
    
    ##Function to check if all videos processed have 3000 frames ##
    ## Returns a list of incomplete videos ##
    def verifyDataset(self, frames_data_path):
        print("Verifying dataset integrity")
        incomplete = []
        folder_list = os.listdir(frames_data_path)
        for folder in tqdm(folder_list) :    
            folder_path = os.path.join(frames_data_path,folder)
            num_frames = len(os.listdir(folder_path))
            if num_frames != 3000:
                incomplete.append(folder)
        return incomplete ,folder_list
    
    ## Funtion to create data vectors 
    ## X of shape (samples, time_step, height , width, channels) 
    ## and Y of shape (samples,5)
    

    def makeVector (self, train_data_path, test_data_path, labels_path,  batch_size =32, time_steps =5 , img_size = (300,215,3) ):
        
        train_data = tf.keras.preprocessing.image_dataset_from_directory(train_data_path, labels= None,
                                                            class_names=None, color_mode='rgb', batch_size=batch_size, image_size=img_size, 
                                                            shuffle=False, seed=None, validation_split=None)
        #train_labels = getLabel(labels_path)
        return train_data
    #def getBatches
    #def getLabel(labels_path):
        
     #   for label in labels_path 
    