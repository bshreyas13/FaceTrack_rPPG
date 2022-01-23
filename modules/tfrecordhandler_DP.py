#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 03 12:52:11 2021

@author: bshreyas

Based on tfrecords for video sequences written by David Molony
Original article with code can be found here: https://dmolony3.github.io/Working%20with%20image%20sequences.html

"""


####################################################################
## Dataset handler for Deep phys                                  ##
## This module has utils to handle video datasets using tfrecords ##
## Util include tfrecord writer parser and necessary helper tools ##
## as required by the two models and returns a batch of X, Y      ##
####################################################################

import os 
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
from natsort import natsorted
import random
import pathlib
import glob
import math
from PIL import Image
import numpy as np 
from tqdm import tqdm
from modules.preprocessor import Preprocessor
from modules.videodatasethandler import VideoDatasetHandler


##################
## Writer Class ##
##################
class TFRWriter():
    
    def __init__(self, img_size):
        self.img_size = img_size
        

    ## Function to take in extracted video frames and create a file list with img path,label ##
    ## data_path : the path to directory with folder of extracted frames (~/Dataset/Roi/Nd)## 
    ## labels_path : path to per video label file (sXX_trial ) 
    ## txt_files_path : Path to save directory of txt_files ##
    def makeFiletxt(self,roi_path,nd_path,in_data, labels_path, txt_files_path):
        p = Preprocessor()
        #in_data = os.listdir(data_path)
        for vidname in in_data:
            filename = os.path.join(txt_files_path,vidname) + '.txt'
            if os.path.isfile(filename):
                os.remove(filename)
            file = open(filename, 'a')           
            frames_roi = natsorted(os.listdir(os.path.join(roi_path,vidname)))
            vid_labels = p.loadData(os.path.join(labels_path,vidname+'.dat'))
            for idx, img in enumerate(frames_roi):
                file.write(os.path.abspath(os.path.join(roi_path,vidname,img)) + " ## " + os.path.abspath(os.path.join(nd_path,vidname,img))+" ## {}\n".format(vid_labels[idx]))
            file.close()
   

    ## Function to shuffle the dataset by trails ##
    ## txt_file_path : the path to txt files with ##
    def makeShuffledDict(self,txt_file_path):
        files = os.listdir(txt_file_path)
        files = [name for name in files if '.txt' in name]
        file_dict = {}
        
        for file in files :
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
        roi_list = [name.split(' ## ')[0] for name in data]
        nd_list = [name.split(' ## ')[1] for name in data]
        label_list = [float(name.split(' ## ')[2]) for name in data] 
        return roi_list,nd_list, label_list
        
        
    ##Function takes a list of images and returns the list in bytes###        
    def getImgBytes(self,img):       
        
        # print(img)
        image = cv2.imread(img)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # # image = Image.open(os.path.join(directory, image))
        # # image = np.asarray(image)
        size = (self.img_size[0],self.img_size[1])
        image = cv2.resize(image, size)       
        # image_bytes = image.tostring()
        image_bytes = cv2.imencode(".jpg", image)[1].tostring()
        image_bytes = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
        
        return image_bytes
    
    ##Function takes a list of labels and returns the list in floa64 ##
    def getLabelBytes(self, label):
        
        label = float(label)
        label_int = tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
            
        return label_int
    
    
    ## Function makes tfrecords of the dataset given batch size and timesteps ##
    ## roi_path: path to appearance stream data ##
    ## nd_path: path to motion stream data ##
    ## txt_files_path: path to txt files with image path , labels ##  
    ## tfrecord_path: path to TFRecord files ##
    ## file_list: list of filenames to produce batches of data from ##
    ## batch_size: defaults to 10 ##
    ## split: train / val / test 
    ## writes a output tfrecord file in the given path ##
    def writeTFRecords(self, roi_path,nd_path,txt_files_path, tfrecord_path, file_list, batch_size,split,timesteps=5):
        # Check split path and mkdir if not present 
        split_path= pathlib.Path(os.path.join(tfrecord_path,split))
        split_path.mkdir(parents=True,exist_ok=True)
        
        print("Number of Videos in {} set: {}".format(split,len(file_list)))
        
        img_size = self.img_size
        ## Calculate shards        
        num_vids = len(file_list)
        
        ## each video is split into 2 shards to keep the size of each tf record under 200 MB
        ## this can be genralized in fututre iterations
        num_shards = num_vids * 2 
        print("Number of TfRecord shards in {} set:{}".format(split,num_shards))
        
        ## Track file count
        file_count = 0
        ## Iterate over number of shards
        ## For each shard write half of total (motion_image,appearance_image) ,(label) pairs
        for shard_no in tqdm(range(num_shards),desc="{} tfrecord in progress".format(split)):
                     
            # Initialize writer
            tfrecord_name = os.path.join(split_path.as_posix(),split +'_'+ str(shard_no)+'.tfrecord')
            writer = tf.io.TFRecordWriter(tfrecord_name)
            
            ## file_count increases by 1 for every 2 shards
            if shard_no != 0 and shard_no % 2 == 0:
                file_count += 1 
            
            ## get roi, nd, label lists each first file in file_list 
            roi_list, nd_list, label_list = self.readFileList(txt_files_path,file_list[file_count][0])
            
            ## Track frame_count
            curr_frame_count = 0
            max_shard_size = len(roi_list)//2
           
            ## Write (motion_image,appearance_image) ,(label) pairs into shard
            while curr_frame_count < max_shard_size :
               
                ## Count index based on file_count
                index = (shard_no % 2) * max_shard_size + curr_frame_count  
                
                images_roi = self.getImgBytes(roi_list[index])
                images_nd = self.getImgBytes(nd_list[index])    
                labels = self.getLabelBytes(label_list[index])
        
                sub_trial = os.path.basename(roi_list[0])
                vidname = sub_trial.split('_f')[0]
        
                im_height = tf.train.Feature(int64_list=tf.train.Int64List(value=[img_size[0]]))
                im_width = tf.train.Feature(int64_list=tf.train.Int64List(value=[img_size[1]]))
                im_depth = tf.train.Feature(int64_list=tf.train.Int64List(value=[img_size[2]]))
                im_name = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(vidname)]))
            
                frames_inseq = roi_list[index].split('_')[-1].split('.jpg')[0]
                frames_inseq = tf.train.Feature(bytes_list=tf.train.BytesList(value =[str.encode(frames_inseq)]))
        
                feature_dict = {'Motion': images_nd,'Appearance':images_roi, 'Labels': labels,'height': im_height, 'width': im_width, 'depth': im_depth, 'name': im_name, 'frames': frames_inseq}
                example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                writer.write(example.SerializeToString())
                    
                curr_frame_count += 1
                
            writer.close()
            
  
##################
## Reader Class ##
##################
class TFRReader():
    
    def __init__(self, batch_size, sequence_length):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
    
    ## Beta feature to add rotation to the entire sequence of images ## 
    ## Not tested ##
    def rotateInputs(self, image, label):
        rot_angle = tf.random.uniform([], minval=0, maxval=360, dtype=tf.float32)
        rot_angle = rot_angle * math.pi / 180
        
        motion_image , appearance_image = image
        motion_image = tfa.image.rotate(motion_image, rot_angle, interpolation = 'BILINEAR')
        appearance_image = tfa.image.rotate(appearance_image, rot_angle, interpolation = 'BILINEAR')
        return (motion_image, appearance_image), (label)
    

    def normalize(img):
        
        return img/255
    
    ## Parses a single example and returns inputs (motion, appearance),y,im_name, frames ##
    ## This method is used to view samples of the dataset ##
    def parseExampleToView(self, sequence_example):
        
        sequence_features = {'Motion': tf.io.FixedLenFeature([], dtype=tf.string),
                             'Appearance': tf.io.FixedLenFeature([], dtype=tf.string),
                          'Labels': tf.io.FixedLenFeature([], dtype=tf.float32),
                          'height': tf.io.FixedLenFeature([], dtype=tf.int64),
                          'width': tf.io.FixedLenFeature([], dtype=tf.int64),
                          'depth': tf.io.FixedLenFeature([], dtype=tf.int64),
                            'name': tf.io.FixedLenFeature([], dtype=tf.string),
                             'frames': tf.io.FixedLenFeature([], dtype=tf.string)}

   
        parsed_ex = tf.io.parse_single_example(
            sequence_example, features=sequence_features)

        # get features context
        im_height = tf.cast(parsed_ex['height'], dtype = tf.int32)
        im_width = tf.cast(parsed_ex['width'], dtype = tf.int32)
        im_depth = tf.cast(parsed_ex['depth'], dtype = tf.int32)
        im_name = parsed_ex['name']
        frames = parsed_ex['frames']

        # decode image
        motion_image = tf.io.decode_jpeg(parsed_ex['Motion'], channels=3)
        motion_image = tf.reshape(motion_image, shape=(im_height, im_width, im_depth))
        
        appearance_image = tf.io.decode_jpeg(parsed_ex['Appearance'], channels=3)
        appearance_image = tf.reshape(appearance_image, shape=(im_height, im_width, im_depth))
        
        label = tf.cast(parsed_ex['Labels'], dtype = tf.float32)
        
        return (motion_image, appearance_image), label, im_name, frames
    
    
    ## Parser for using with model , produces (motion,appearance),(labels) ##
    ## This method is used to train model ##
    def parseExample(self, sequence_example):
        
        sequence_features = {'Motion': tf.io.FixedLenFeature([], dtype=tf.string),
                             'Appearance': tf.io.FixedLenFeature([], dtype=tf.string),
                          'Labels': tf.io.FixedLenFeature([], dtype=tf.float32),
                          'height': tf.io.FixedLenFeature([], dtype=tf.int64),
                          'width': tf.io.FixedLenFeature([], dtype=tf.int64),
                          'depth': tf.io.FixedLenFeature([], dtype=tf.int64),
                            'name': tf.io.FixedLenFeature([], dtype=tf.string),
                             'frames': tf.io.FixedLenFeature([], dtype=tf.string)}

   
        parsed_ex = tf.io.parse_single_example(
            sequence_example, features=sequence_features)

        # get features context
        im_height = tf.cast(parsed_ex['height'], dtype = tf.int32)
        im_width = tf.cast(parsed_ex['width'], dtype = tf.int32)
        im_depth = tf.cast(parsed_ex['depth'], dtype = tf.int32)


        # decode image
        motion_image = tf.io.decode_jpeg(parsed_ex['Motion'], channels=3)
        #motion_image = tf.reshape(motion_image, shape=(im_height, im_width, im_depth))
        
        appearance_image = tf.io.decode_jpeg(parsed_ex['Appearance'], channels=3)
        #appearance_image = tf.reshape(appearance_image, shape=(im_height, im_width, im_depth))
        
        label = tf.cast(parsed_ex['Labels'], dtype = tf.float32)
        
        return (motion_image/255, appearance_image/255), (label)
    
    
    
    ## Reads TFRecord and produces batch objects for training ##
    def getBatch(self, dirname, subset,to_view = False, rotate=0):
        
        ## TF Performance Configuration
        try:
            AUTOTUNE = tf.data.AUTOTUNE     
        except:
            AUTOTUNE = tf.data.experimental.AUTOTUNE 
        
        print(dirname)  
        files = natsorted(glob.glob(dirname + '/*.tfrecord'))
        print("No of shards of data found:",len(files))
        vdh = VideoDatasetHandler()
        in_data = vdh.getSubset(files,subset)
        print("Using {}% of the TFRecord shards: {} ".format(subset*100,len(in_data)))
        dataset = tf.data.TFRecordDataset(in_data)
        
        if to_view == True:            
            dataset = dataset.map(self.parseExampleToView, num_parallel_calls=AUTOTUNE)
        else:
            dataset = dataset.map(self.parseExample, num_parallel_calls = AUTOTUNE)
        if rotate == 1:
            dataset = dataset.map(self.rotateInputs, num_parallel_calls=AUTOTUNE)
        

        dataset = dataset.batch(self.batch_size)
        
        return dataset