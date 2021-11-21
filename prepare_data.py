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

def dataGenerator ( model,in_data, data_path, labels_path,  batch_size =50, time_steps = 5 , img_size = (300,215,3)):
       
        
        if model == 'DeepPhys' :
            batch_imgs , batch_labels = [] , []
            for folder in in_data :
                path = os.path.join(data_path,folder)
                imgs = natsorted(os.listdir(path))
                label_file = vdh.getLabelFile(labels_path,folder)
                b_count = 0
                for idx, image in enumerate(imgs) :
                    while b_count < batch_size :
                        img = cv2.imread(os.path.join(data_path,folder,image))
                        label = vdh.getLabel(label_file,idx)
                        batch_imgs.append(img)
                        batch_labels.append(label)
                        b_count += 1
                    yield np.array(batch_imgs) , np.array(batch_labels)                 
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
n = dataGenerator('DeepPhys', in_data,roi ,labels_save_path)
for k in range (1):    
    i,y = next(n)
    print(i.shape)
    print(y)