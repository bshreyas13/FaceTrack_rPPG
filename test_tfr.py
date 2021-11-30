#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:24:53 2021

@author: bshreyas
"""
from modules.tfrecordhandler import TFRHandler
import os
import numpy as np
import tensorflow as tf

data_path = "../Dataset/Roi"
txt_files_path = "../Dataset/test"
labels_path = "../../Labels"
in_data = ["s01_trial10"]

tfr = TFRHandler()
tfr.makeFiletxt(data_path,in_data,labels_path,txt_files_path)
txt = os.listdir(txt_files_path)
a,b = tfr.readFileList(txt_files_path,txt[1])
print(a[-1])
file_list= tfr.makeShuffledDict(txt_files_path)
batch_size = 1
timesteps=5
split = "train"
img_size = (300,215,3)
writer = tf.io.TFRecordWriter(split + '.tfrecord')
## Testing tf record writer to write the data in proper timesteps ans batched manner ## 
for i in range(0, len(file_list), batch_size):
        # read files
        j = i
        num_files = 1e6
        full_batch_image_list = []
        full_batch_label_list = []
        while j < i + batch_size:
            # get maximum number of files in each dataset
            num_files = min(num_files, file_list[j][1])
            image_list, label_list = tfr.readFileList(txt_files_path, file_list[j][0])
            j += 1
            full_batch_image_list.append(image_list)
            full_batch_label_list.append(label_list)
# iterate over timesteps and add each batch 
num_seqs = num_files//timesteps
current_timestep = 0
while current_timestep < timesteps*num_seqs: 
            for l in range(batch_size):
                t = full_batch_image_list[l][current_timestep:current_timestep+timesteps]
                image_bytes_list = tfr.getImgSeqBytes(data_path, full_batch_image_list[l][current_timestep:current_timestep+timesteps])       
                label_int_list = tfr.getLabelSeqBytes(full_batch_label_list[l][current_timestep:current_timestep+timesteps])
           
            current_timestep += timesteps
print(len(image_bytes_list))