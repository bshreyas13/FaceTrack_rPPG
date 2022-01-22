#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 02:51:40 2021

@author: bshreyas
"""

from modules.videodatasethandler import VideoDatasetHandler
import os

## Function gets subset of data , splits data to obtain train , val, test sets ##
## Return: 3 lists of video_folder names sXX_trialXX ##
def getSets( motion_path, subset=0.01 , val_split=0.1, test_split=0.2):
    vdh = VideoDatasetHandler()
    all_videos = os.listdir(motion_path)
    in_data = vdh.getSubset(all_videos,subset)    
    train_set, val_set, test_set = vdh.splitData(in_data,val_split, test_split)  
    return train_set, val_set, test_set 

## Function gets sfixes label files names with missing zeros to proper format sXX_trialXX.dat##
def fixLabelFilenames(labels_path):
     for label_file in os.listdir(labels_path):
        trial_num = label_file.split('trial')[-1].split('.')[0]
        if len(trial_num)==1 :
            new_name = label_file.split('trial')[0]+'trial'+'0'+trial_num+'.dat'
            # print(new_name)
            os.rename(os.path.join(labels_path,label_file),os.path.join(labels_path,new_name))
 
def createLists(model,roi_path,nd_path,labels_path, train_set, val_set, test_set, txt_files_paths, write_txt_files=False):
    
    if model == "FaceTrack_rPPG":
        from modules.tfrecordhandler_FTR import TFRWriter
        
    else :
        from modules.tfrecordhandler_DP import TFRWriter
        
    tfwrite = TFRWriter()   
    train_txt_path = txt_files_paths[0]
    val_txt_path = txt_files_paths[1]
    test_txt_path = txt_files_paths[2]
    if write_txt_files == True:
        print("Creating Train List")       
        tfwrite.makeFiletxt(roi_path,nd_path, train_set,labels_path,train_txt_path) ## Write txt file with train video    
        print("Creating Val List")
        tfwrite.makeFiletxt(roi_path,nd_path, val_set,labels_path,val_txt_path) ## Write txt file with val video    
        print("Creating Test List")
        tfwrite.makeFiletxt(roi_path,nd_path, test_set,labels_path,test_txt_path) ## Write txt file with test video    
    
    train_list= tfwrite.makeShuffledDict(train_txt_path)
    val_list= tfwrite.makeShuffledDict(val_txt_path)
    test_list= tfwrite.makeShuffledDict(test_txt_path)
    
    return train_list, val_list,test_list

## Function takes available videos to obtian a subset of given size ##
## Then splits them to train,validation and test sets ##
## roi_path: path to Apperance frame ##
## nd_path : path to motion frames ##
## labels_path : path to label files ##
## txt_files_paths : list of paths tp train, val and test txt files to be saved and accessed ##
## batch_size : defaults to 10 ## 
## subset : subset of data to be used. Input between 0 and 1 ##
## val_split , test_split : input between 0 and 1 ##
## write_txt_files: Flag set to False to skip creation of txt files for tfrecord creation ##
## create_tfrecord : Flag set to False to skip creation of tfrecord files ##
def getDatasets(model,roi_path,nd_path,labels_path,txt_files_paths,tfrecord_path,img_size, batch_size=10, timesteps=5, subset=0.25, subset_read = 1, val_split = 0.1 , test_split =0.2, write_txt_files=False, create_tfrecord=False, rot = 0):
    
    
    if model == "FaceTrack_rPPG":
        print("Imported Tf record handler for {}".format(model))
        from modules.tfrecordhandler_FTR import TFRWriter
        from modules.tfrecordhandler_FTR import TFRReader
        
    else  :
        print("Imported Tf record handler for DeepPhys")
        from modules.tfrecordhandler_DP import TFRWriter
        from modules.tfrecordhandler_DP import TFRReader
    
    tfwrite = TFRWriter()   
    ## get subset and split data
    train_set, val_set, test_set = getSets(nd_path,subset,val_split,test_split)
    
    ## get Lists of files for each set
    train_list, val_list,test_list = createLists(model,roi_path,nd_path,labels_path, train_set, val_set, test_set, txt_files_paths,write_txt_files)
    
    print("No of train videos:",len(train_list))
    print("No of validation videos:",len(val_list))
    print("No of test videos:",len(test_list))
        
    if create_tfrecord == True:
            print("In Progress: Writing tfrecords")
            ## Make Train.tfrecord 
            tfwrite.writeTFRecords(roi_path,nd_path, txt_files_paths[0], tfrecord_path, train_list, batch_size,'Train',timesteps,img_size=img_size)
            
            ## Make val.tfrecord 
            tfwrite.writeTFRecords(roi_path,nd_path, txt_files_paths[1], tfrecord_path, val_list, batch_size,'Val',timesteps,img_size=img_size)
            
            ## Make test.tfrecord 
            tfwrite.writeTFRecords(roi_path,nd_path, txt_files_paths[2], tfrecord_path, test_list, batch_size,'Test',timesteps,img_size=img_size)
    try:
  
        train_tfrpath = os.path.join(tfrecord_path,'Train')
        # get batches from dAatset iterator
        train_data = TFRReader(batch_size, timesteps)
        train_batch = train_data.getBatch(train_tfrpath, subset_read, False, rotate = rot)
        
        val_tfrpath = os.path.join(tfrecord_path,'Val')
        # get batches from dAatset iterator
        val_data = TFRReader(batch_size, timesteps)
        val_batch = val_data.getBatch(val_tfrpath, subset_read, False, rotate = 0)
    
        test_tfrpath = os.path.join(tfrecord_path,'Test')
        # get batches from dAatset iterator
        test_data = TFRReader(batch_size, timesteps)
        test_batch = test_data.getBatch(test_tfrpath, subset_read, False, rotate = 0)
    
        return train_batch, val_batch, test_batch
    
    except:
        print("Check tfrecords path")
        print("If TFRecords are not created enable creation by setting create_tfrecord = True")