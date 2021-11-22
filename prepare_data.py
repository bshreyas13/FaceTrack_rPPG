# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 00:44:51 2021

@author: local_test
"""
import os
import pathlib
import cv2
import numpy as np
from natsort import natsorted
import re
import tensorflow as tf
from modules.videodatasethandler import VideoDatasetHandler
from modules.preprocessor import Preprocessor 

## The script gets subset of data , splits data to obtain train , val, test sets ##
## Then calls datagen to obtain samples ##
## calls train_test_plot to train and evaluate model ##

import os
from modules.videodatasethandler import VideoDatasetHandler
from modules.preprocessor import Preprocessor

def dataGenerator ( model,in_data, data_path, labels_path,  batch_size =50, time_steps = 5 , img_size = (300,215,3)):
        
        if model == 'DeepPhys' :        
            for folder in in_data :
                path = os.path.join(data_path,folder)
                imgs = natsorted(os.listdir(path))
                label_file = vdh.getLabelFile(model,labels_path,folder)
                video_file = vdh.getImageStack(model,data_path,folder, imgs)
                l = len(imgs)
                for idx in range(0,l,batch_size) :
                        batch_X = video_file[idx:min(idx+batch_size,l)]
                        batch_Y = label_file[idx:min(idx+batch_size,l)]
                        yield np.array(batch_X) , np.array(batch_Y)
                    
        elif model == 'FaceTrack_rPPG' :
            for folder in in_data :
                path = os.path.join(data_path,folder)
                imgs = natsorted(os.listdir(path))
                label_file = vdh.getLabelFile(model,labels_path,folder)
                video_file = vdh.getImageStack(model,data_path,folder, imgs)
                l = len(imgs)
                for idx in range(0,l,batch_size) :
                        batch_X = video_file[idx:min(idx+batch_size,l)]
                        batch_Y = label_file[idx:min(idx+batch_size,l)]
                        yield np.array(batch_X) , np.array(batch_Y)
vdh = VideoDatasetHandler()
p = Preprocessor()

in_data = ['s11_trial27']
labels_path =  (os.path.join(os.path.dirname(os.getcwd()),'Labels'))
roi = os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Roi')
datagen = dataGenerator('DeepPhys', in_data, roi ,labels_path)

x,y = next(datagen)
print(x.shape)
print(y.shape)
    
