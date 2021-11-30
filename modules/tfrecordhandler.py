#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:52:11 2021

@author: bshreyas
"""
import os 
import cv2
import tensorflow as tf
import operator
from natsort import natsorted
import random
from modules.preprocessor import Preprocessor

####################################################################
## This class has utils to handle video datasets using tfrecords  ##
## Util include tfrecord writer parser abd necessary tools        ##
## as required by the two models and returns a batch of X, Y      ##
####################################################################

class TFRHandler():
    
    ## Funtion to take in extracted video frames and create a file list with img path,label ##
    ## data_path : the path to directory with folder of extracted frames (~/Dataset/Roi/Nd)## 
    ## labels_path : path to per video label file (sXX_trial ) 
    ## txt_files_path : Path to save directory of txt_files ##
    def makeFiletxt(self,data_path, in_data, labels_path, txt_files_path):
        p = Preprocessor()
        #data = os.listdir(data_path)
        for vidname in in_data:
            filename = os.path.join(txt_files_path,vidname) + '.txt'
            if os.path.isfile(filename):
                continue
            file = open(filename, 'a')           
            frames = natsorted(os.listdir(os.path.join(data_path,vidname)))
            vid_labels = p.loadData(os.path.join(labels_path,vidname+'.dat'))
            for idx, img in enumerate(frames):
                file.write(os.path.abspath(os.path.join(data_path,vidname,img)) + " {}\n".format(vid_labels[idx]))
            file.close()
   
    ## Function to shuffle the dataset by trails ##
    ## txt_file_path : the path to txt files with ##
    ## yet to add subsampling feature ##
    def makeShuffledDict(self,txt_file_path):
        files = os.listdir(txt_file_path)
        files = [name for name in files if '.txt' in name]
        file_dict = {}
        
        for file in files:
            f = open(os.path.join(txt_file_path, file), 'r')
            num_files = len(f.read().splitlines())
            file_dict[file] = num_files
            f.close()
        shuffled = list(file_dict.items())
        random.shuffle(shuffled)
       
        return shuffled
    
    ## Function to read txt files with img_path and corresponding labels ##
    ## directory : the path to directory with txt files (~/Dataset/txt_files/Roi)## 
    ## file : individual filename (sXX_trial ) 
    ## txt_files_path : Path to save directory of txt_files ##
    def readFileList(self,directory, file):
        f = open(os.path.join(directory, file), 'r')
        data=f.read().splitlines()
        f.close()
        image_list = [name.split(' ')[0] for name in data]
        label_list = [float(name.split(' ')[1]) for name in data] 
        return image_list, label_list
        
        
    ##Function takes a list of images and returns the list in bytes###        
    def getImgSeqBytes(self,directory, image_list,img_size =(300,215)):       
        image_bytes_seq = []
        for image in image_list:
            image = cv2.imread(os.path.join(directory, image))
            image = cv2.resize(image,img_size)
            image_bytes = image.tostring()
            image_bytes = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            image_bytes_seq.append(image_bytes)

        return image_bytes_seq
    
    ##Function takes a list of labels and returns the list in floa64 ##
    def getLabelSeqBytes(self, label_list):
        label_seq = []
        for label in label_list:
            label = float(label)
            label_int = tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
            label_seq.append(label_int)
        return label_seq
