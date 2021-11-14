# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 23:31:07 2021

@author: bshreyas
"""

import cv2
import pathlib
import os

class MakeVideoDataset:
    
    def __init__(self,video_path,dataset_path):
      self.video_path = video_path
      self.dataset_path = dataset_path
      
    def prepareDataset(self):
        
        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(3)) 
        frame_height = int(cap.get(4)) 
        size = (frame_width, frame_height)
        
        ## Paths setup
        source_path = pathlib.PurePath(self.video_path)
        filename = source_path.name  
        
        save_path = os.path.join(self.dataset_path,filename)
        save_path.mkdir(parents=True,exist_ok=True)