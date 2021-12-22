#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 23:21:34 2021

@author: shreyas
"""
import tensorflow as tf
import os 
import pathlib
import numpy as np
import cv2

roi_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset','Roi'))               
nd_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset','Nd'))
labels_path =  pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Labels'))

video_list = os.listdir(roi_path)
video_data = []
num_frames = 5
vid_path = os.path.join(roi_path,video_list[0])
imgs = os.listdir(vid_path)
for i, img in enumerate(imgs):
    video_data.append(cv2.imread(os.path.join(vid_path,img)))
    if i == 4:
        break
video_data = np.stack(video_data)

img_bytes = [tf.io.encode_jpeg(frame, format='rgb') for frame in video_data]

print(type(img_bytes))

for byte in img_bytes :
    print(len(byte))