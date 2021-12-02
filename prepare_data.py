#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 02:51:40 2021

@author: bshreyas
"""

import tensorflow as tf
from modules.videodatasethandler import VideoDatasetHandler
import os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from modules.tfrecordhandler import TFRWriter
from modules.tfrecordhandler import TFRReader
import argparse as ap

## Function gets subset of data , splits data to obtain train , val, test sets ##
## Return: 3 lists of video_folder names sXX_trialXX ##
def getSets( motion_path, subset=0.01 , val_split=0.1, test_split=0.2):
    vdh = VideoDatasetHandler()
    all_videos = os.listdir(motion_path)
    in_data = vdh.getSubset(all_videos,subset)    
    train_set, val_set, test_set = vdh.splitData(in_data,val_split, test_split)  
    return train_set, val_set, test_set 


def getDataset(roi_path,nd_path,in_data,labels_path,txt_files_path):
    
    tfwrite = TFRWriter()
    tx_file = tfwrite.makeFiletxt(roi_path,nd_path, in_data,labels_path,txt_files_path) ## roi and nd together
    txt = os.listdir(txt_files_path)
    
    file_list= tfwrite.makeShuffledDict(txt_files_path)
