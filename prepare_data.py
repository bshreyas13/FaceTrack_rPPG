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

vdh = VideoDatasetHandler()
p = Preprocessor()

in_data = ['s11_trial27']
labels_path =  (os.path.join(os.path.dirname(os.getcwd()),'Labels'))
roi = os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Roi')
datagen = vdh.dataGenerator('FaceTrack_rPPG', in_data,roi ,labels_path)
    
x,y = next(datagen)
print(x.shape)
print(y.shape)
    
