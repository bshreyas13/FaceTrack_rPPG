#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:52:11 2021

@author: bshreyas
"""
import os 
import cv2
import tensorflow as tf
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
    def makeFiletxt(self,roi_path,nd_path,in_data, labels_path, txt_files_path):
        p = Preprocessor()
        #in_data = os.listdir(data_path)
        for vidname in in_data:
            filename = os.path.join(txt_files_path,vidname) + '.txt'
            if os.path.isfile(filename):
                os.remove(filename)
            file = open(filename, 'a')           
            frames_roi = natsorted(os.listdir(os.path.join(roi_path,vidname)))
            frames_roi = natsorted(os.listdir(os.path.join(nd_path,vidname)))
            vid_labels = p.loadData(os.path.join(labels_path,vidname+'.dat'))
            for idx, img in enumerate(frames_roi):
                file.write(os.path.abspath(os.path.join(roi_path,vidname,img)) + " " + os.path.join(nd_path,vidname,img)+" {}\n".format(vid_labels[idx]))
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
        roi_list = [name.split(' ')[0] for name in data]
        nd_list = [name.split(' ')[1] for name in data]
        label_list = [float(name.split(' ')[2]) for name in data] 
        return roi_list,nd_list, label_list
        
        
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
    
    def getTFRecords(self, roi_path,nd_path,tfrecord_path, txt_files_path, file_list, batch_size,split,timesteps=5, img_size=(300,215,3)):
        ## Initialize writer
        writer = tf.io.TFRecordWriter(os.path.join(tfrecord_path.as_posix(), split + '.tfrecord'))

        ## Testing tf record writer to write the data in proper timesteps ans batched manner ## 
        ## Iterate through dict of shuffled videos
        for i in range(0,len(file_list),batch_size):
            # read files
            j = i
            num_files = 3000
            full_batch_roi_list = []
            full_batch_nd_list = []
            full_batch_label_list = []
            while j < i + batch_size:
                # get maximum number of files in each dataset
                num_files = min(num_files, file_list[j][1])
                roi_list, nd_list, label_list = self.readFileList(txt_files_path, file_list[j][0])
                j += 1
                full_batch_roi_list.append(roi_list)
                full_batch_nd_list.append(nd_list)
                full_batch_label_list.append(label_list)
        # iterate over timesteps and add each batch 
        num_seqs = num_files//timesteps
        current_timestep = 0
        while current_timestep < timesteps*num_seqs: 
                for l in range(batch_size):
                
                    roi_bytes_list = self.getImgSeqBytes(roi_path, full_batch_roi_list[l][current_timestep:current_timestep+timesteps])
                    nd_bytes_list = self.getImgSeqBytes(nd_path, full_batch_nd_list[l][current_timestep:current_timestep+timesteps])    
                    label_bytes_list = self.getLabelSeqBytes(full_batch_label_list[l][current_timestep:current_timestep+timesteps])
                
                    sub_trial = os.path.basename(full_batch_roi_list[l][0])
                    vidname = sub_trial.split('_f')[0]
                
                    images_roi = tf.train.FeatureList(feature=roi_bytes_list)
                    images_nd = tf.train.FeatureList(feature=nd_bytes_list)
                    labels = tf.train.FeatureList(feature=label_bytes_list)
                
                    im_length = tf.train.Feature(int64_list=tf.train.Int64List(value=[len(roi_bytes_list)]))
                    im_height = tf.train.Feature(int64_list=tf.train.Int64List(value=[img_size[0]]))
                    im_width = tf.train.Feature(int64_list=tf.train.Int64List(value=[img_size[1]]))
                    im_depth = tf.train.Feature(int64_list=tf.train.Int64List(value=[img_size[2]]))
                    im_name = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(vidname)]))
                    
                    frames_inseq = list(map(lambda x: x.split('_')[-1].split('.jpg')[0], full_batch_roi_list[l][current_timestep:current_timestep+timesteps]))
                    frames_inseq = "".join(frames_inseq)
                    frames_inseq = tf.train.Feature(bytes_list=tf.train.BytesList(value =[str.encode(frames_inseq)]))
                
                    # create a dictionary
                    sequence_dict = {'Motion': images_nd,'Appeareance':images_roi, 'Labels': labels}
                    context_dict = {'length': im_length, 'height': im_height, 'width': im_width, 'depth': im_depth, 'name': im_name, 'frames': frames_inseq}

                    sequence_context = tf.train.Features(feature=context_dict)
                    # now create a list of feature lists contained within dictionary
                    sequence_list = tf.train.FeatureLists(feature_list=sequence_dict)

                    example = tf.train.SequenceExample(context=sequence_context, feature_lists=sequence_list)
                    writer.write(example.SerializeToString())

                current_timestep += timesteps
        writer.close()

