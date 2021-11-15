# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:06:53 2021

@author: local_test
"""
import argparse as ap
from tqdm import tqdm
import time 
import os
import pathlib
import numpy as np
from preprocessor import Preprocessor
from videodatasethandler import VideoDatasetHandler

if __name__ == '__main__':
    
    parser = ap.ArgumentParser()
    parser.add_argument("-ds","--data_source", required = True , help = "path to video source directory with unsplit dataset")
    parser.add_argument("-lp","--label_path", required = True , help = "path to directory with labels .mat signals")
    
    args = vars(parser.parse_args())
    
    data_path = args['data_source']
    label_path = args['label_path']
    
    #####################################################################
    ## Script to process all videos in the Deap folder                 ##
    ## Each subject has 40 trials of 1 minute each                     ##
    ## There is one label file with 40 signals corresponding to trials ##
    #####################################################################
    
    ## Intialize preprocessor 
    f = Preprocessor()
    
    ## Get Roi for all videos ##
    start = time.time()
    
    ## Resize roi videos to standardize dims 
    rsz_dim = (300,215)
    
    roi_save_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Roi_Videos'))           
    roi_save_path.mkdir(parents=True,exist_ok=True)
    
    nd_save_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'ND_Videos'))
    nd_save_path.mkdir(parents=True,exist_ok=True)
    
    labels_save_path =  pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Labels'))
    labels_save_path.mkdir(parents=True,exist_ok=True)
    
    dataset_save_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Roi'))
    dataset_save_path.mkdir(parents=True,exist_ok=True)

    dataset_save_path_nd = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Nd'))
    dataset_save_path_nd.mkdir(parents=True,exist_ok=True)
    
    #Check progress
    processed_roi = os.listdir(roi_save_path)
    processed_nd = os.listdir(nd_save_path)
    ## First Track Face and Extract Roi for all videos 
    print("Strating Roi Extraction.")
    data_folders = os.listdir(data_path)
  
    
    ###To skip specific files from log ##
    skip_list = ['s11_trial15.avi',
                 's15_trial12.avi',
                 's15_trial16.avi',
                 's15_trial23.avi',
                 's12_trial14.avi']
    for folder in tqdm(data_folders):
        video_list = os.listdir(os.path.join(data_path,folder))
       
        for video_name in video_list :
            if video_name in processed_roi or video_name in skip_list :
                continue
            video = os.path.join(data_path,folder,video_name)
            with open('log_processed.txt', 'a') as file:
                        file.write("%s\n" %video_name )
            img = f.getRoi(video, rsz_dim, roi_save_path, dataset_save_path)
    
    ## Get normalized difference frame  
    roi_vids = os.listdir(roi_save_path.as_posix())
    for vid_name in tqdm(roi_vids):
        if vid_name not in processed_nd:
            vid = os.path.join (roi_save_path.as_posix(), vid_name)
            n_d = f.getNormalizedDifference( vid ,nd_save_path,dataset_save_path_nd)
    
    end = time.time()
    print("All videos processed. Roi and Difference frames saved")
   
    #cv2.imwrite('test.png',img)
    #cv2.imwrite('test_nd.png',n_d)
    
    ## Process Labels (PPG signals)
    print("Downsampling and Preparing labels/trial")
    
    label_files = os.listdir(label_path)
    for labels in tqdm(label_files):
        if labels.split('.')[-1] == 'mat' and len(labels.split(' '))==1 :
            #print(labels)
            labels_source = os.path.join(label_path,labels)
            y = f.loadLabels(labels_source)
            
            for i in range(len(y)):
                resampled = f.matchIoSr(y[i])
                resampled =np.array(resampled)
                derivative = f.getDerivative(resampled)
                
                save_name = labels.split('_')[0] + '_trial'+str(i+1)+'.dat' 
                save_path = os.path.join(labels_save_path,save_name)
                f.saveData(save_path,derivative)
    
    #f.plotHR(y[-1],128)
    #f.plotHR(resampled,50)
    #t = f.loadData(save_path)
    #f.plotHR(t,50)


    

