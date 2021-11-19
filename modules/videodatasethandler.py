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
    
    ## Funtion to create datat vectors 
    ## X of shape (samples, time_step, height , width, channels) 
    ## and Y of shape () 

    #def makeVector (self, frames_data_path, labels_path time_steps , batch_size):
     
    #def getBatches
    #def getLabel
    