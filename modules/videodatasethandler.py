# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 23:31:07 2021

@author: bshreyas
"""

import cv2
import os
import numpy as np
from tqdm import tqdm
import random
import tensorflow as tf
from natsort import natsorted 
from modules.preprocessor import Preprocessor

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
    
    ## Generator to yield vectors ##
    ## data_path : path to data directory ##
    ## labels_path : path to preprocessed labels directory ##
    ## For DeepPhys ##
    ## For FaceTrack_rPPG ##
    ## X of shape (batch,  time_step, height , width, channels) ##
    ## and Y of shape (batch,5) ##  
    def dataGenerator (self, model,in_data, data_path, labels_path,  batch_size =50, time_steps = 5 , img_size = (300,215,3)):
       
        
        if model == 'DeepPhys' :
            for folder in in_data :
                path = os.path.join(data_path,folder)
                imgs = natsorted(os.listdir(path))
                label_file = self.getLabelFile(labels_path,folder)
                for idx, image in enumerate(imgs) :
                    img = cv2.imread(os.path.join(data_path,folder,image))
                    label = self.getLabel(label_file,idx)
                    yield img , label
                    
        elif model == 'FaceTrack_rPPG' :
            data = []
        
    
    ## Function using reservoir sampling to get a subset of data ##
    ## data:  list of data directory names (sXX_trialXX) ##
    ## subset : percentage of dataset we are considering for the data_subset ##
    ## return :  data_subset (a list of foldernames of format sXX_trialXX) ##   
    def getSubset(self, data, subset=0.01):
        
        num_samples = int(subset * len(data))
        data_subset = []
        for k, video in enumerate(data):
            if k < num_samples:                
                data_subset.append(video) 
            else:              
                i = random.randint(0, k)
                if i < num_samples:
                     data_subset[i] = video
        
        return data_subset
    
    ## Function to split data into train validation and test set ##
    ## data : data:  list of data directory names (sXX_trialXX) ##
    ## return : train,val and test splits from the given data list ##
    def splitData(self,data,val_split =0.1,test_split = 0.2 ):
        
        test_set = self.getSubset(data,test_split)
        
        train_val = []
        for video in data:
            if video in test_set:
                continue
            train_val.append(video)
        
        val_set = self.getSubset(train_val, val_split)
        
        train_set = []
        for video in train_val:
            if video in test_set:
                continue
            train_set.append(video)
        
        return train_set, val_set, test_set
    
    def getImage(self, path ):
        img = cv2.imread(path)
        return img
    def getLabelFile(self,path, vid_name):
        p = Preprocessor()
        label_file = p.loadData(os.path.join(path,vid_name+'.dat'))
        return label_file
    def getLabel(self, label_file, index):
        label = label_file[index]
        return label