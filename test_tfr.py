#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:24:53 2021

@author: bshreyas
"""

import os
import numpy as np
import pathlib
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from modules.tfrecordhandler import TFRWriter
from modules.tfrecordhandler import TFRReader

in_data = ["s01_trial10"]

roi_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset','Roi'))               

nd_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset','Nd'))
labels_path =  pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Labels'))
    
txt_files_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Txt'))
txt_files_path.mkdir(parents=True,exist_ok=True)

tfrecord_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'TFRecords'))
tfrecord_path.mkdir(parents=True,exist_ok=True)

tfwrite = TFRWriter()
roi = tfwrite.makeFiletxt(roi_path,nd_path, in_data,labels_path,txt_files_path) ## roi and nd together
txt = os.listdir(txt_files_path)

file_list= tfwrite.makeShuffledDict(txt_files_path)

test_list = ['/Users/bshreyas/Documents/rPPG/Dataset/Roi/s01_trial10/s01_trial10_f1.jpg',
             '/Users/bshreyas/Documents/rPPG/Dataset/Nd/s01_trial10/s01_trial10_f2.jpg',
             '/Users/bshreyas/Documents/rPPG/Dataset/Roi/s01_trial10/s01_trial10_f3.jpg',
             '/Users/bshreyas/Documents/rPPG/Dataset/Roi/s01_trial10/s01_trial10_f4.jpg', 
             '/Users/bshreyas/Documents/rPPG/Dataset/Roi/s01_trial10/s01_trial10_f5.jpg']
## Batching is done on Video level ==> use small batch size
batch_size = 2
if batch_size > len(in_data):
    batch_size = len(in_data)
timesteps=5
split = "train"
img_size = (215,300,3)

try:
        AUTOTUNE = tf.data.AUTOTUNE     
except:
        AUTOTUNE = tf.data.experimental.AUTOTUNE 

tfwrite.getTFRecords(roi_path,nd_path, txt_files_path, tfrecord_path, file_list, batch_size,'example')

tfrpath = os.path.join(tfrecord_path,'example.tfrecord')
# make a dataset iterator
data = TFRReader(batch_size, timesteps, num_epochs=10)
batch = data.read_batch(tfrpath, 0)


for x_l,x_r,y,name,frame in batch.take(2):    
        print('Appeareance Input Shape:',x_r.shape)
        
        print('Motion Input Shape',x_l.shape)
        print('Output',y.shape)
        print('Video name:',name.numpy())
        
        
        frames = frame.numpy().astype(str)[0].split('f')
        frames= [int(num) for num in frames if num]

        fig = plt.figure(figsize=(12,10))
        idx = 1
        for i in range(0, batch_size):
            print('Displaying frames {}'.format(frames))
            for j in range(0, timesteps):
                # Display the frames along with the label by looking up the dictionary key
                ax = fig.add_subplot(batch_size, timesteps, idx)
                ax.imshow(x_r.numpy()[i, j, : ,: ,:])
                # ax.imshow(x_l.numpy()[i, j, : ,: ,:])
                idx += 1
        plt.show()

