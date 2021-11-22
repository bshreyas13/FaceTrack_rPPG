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
## write getSubset before calling generator 

in_data = ['s11_trial27']
labels_save_path =  (os.path.join(os.path.dirname(os.getcwd()),'Labels'))
roi = os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Roi')
data_path = roi
labels_path = labels_save_path
vdh = VideoDatasetHandler()
p = Preprocessor()
# test = tf.data.Dataset.from_generator( dataGenerator,args = ['DeepPhys', in_data,roi ,labels_save_path],output_types=(tf.int8,tf.float32),output_shapes=((215,300,3), ()))

# dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32, tf.int16))

# for batch, (x,y) in enumerate(test):
  
#     print("Data shape: ", x.shape, y.shape)

a =  p.loadData(os.path.join(labels_save_path,in_data[0]+'.dat'))
n = vdh.dataGenerator('FaceTrack_rPPG', in_data,roi ,labels_save_path)
for k in range (1):    
    i,y = next(n)
    print(i.shape)
    print(y.shape)