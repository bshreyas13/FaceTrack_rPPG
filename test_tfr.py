#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:24:53 2021

@author: bshreyas
"""
from modules.tfrecordhandler import TFRHandler
import os
import numpy as np
import pathlib
import tensorflow as tf


in_data = ["s01_trial10"]

roi_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset','Roi'))               

nd_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset','Nd'))
labels_path =  pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Labels'))
    
txt_files_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Txt'))
txt_files_path.mkdir(parents=True,exist_ok=True)

tfrecord_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'TFRecords'))
tfrecord_path.mkdir(parents=True,exist_ok=True)

tfr = TFRHandler()
roi = tfr.makeFiletxt(roi_path,nd_path, in_data,labels_path,txt_files_path) ## roi and nd together
txt = os.listdir(txt_files_path)

file_list= tfr.makeShuffledDict(txt_files_path)

## Batching is done on Video level ==> use small batch size
batch_size = 2
if batch_size > len(in_data):
    batch_size = len(in_data)
timesteps=5
split = "train"
img_size = (300,215,3)



tfr.getTFRecords(roi_path,nd_path,tfrecord_path, txt_files_path, file_list, batch_size,'example')