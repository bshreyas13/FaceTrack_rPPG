# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 23:31:07 2021

@author: bshreyas
"""

import cv2
import pathlib
import os

class MakeVideoDataset:
    
    def __init__(self,dataset_path,label_path):
      
      self.dataset_path = dataset_path
      self.label_path = label_path
    
    ##Function to check if all videos processed have 3000 frames
    def verifyDataset(self):
        folder_list = os.listdir(self.dataset_path)
        for folder in folder_list :
            
            folder_path = os.path.join(self.dataset_path,folder)
            num_frames = len(os.listdir(folder_path))
            if num_frames != 3000:
                print(folder)
    
        
        