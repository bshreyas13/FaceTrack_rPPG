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

    #####################################################################
    ## Script to process all videos in the Deap folder                 ##
    ## Each subject has 40 trials of 1 minute each                     ##
    ## There is one label file with 40 signals corresponding to trials ##
    #####################################################################
    
if __name__ == '__main__':
    
    parser = ap.ArgumentParser()
    parser.add_argument("-ds","--data_source", required = True , help = "path to video source directory with unsplit dataset")
    parser.add_argument("-lp","--label_path", required = True , help = "path to directory with labels .mat signals")
    
    args = vars(parser.parse_args())
    
    data_path = args['data_source']
    label_path = args['label_path']

    
    ## Intialize preprocessor 
    f = Preprocessor()
    vdh = VideoDatasetHandler()
    
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
    
    ## Check and display progress
    processed_roi = os.listdir(roi_save_path)
    processed_nd = os.listdir(nd_save_path)   
    incomp_processed_frames, comp_processed_frames = vdh.verifyDataset(dataset_save_path)
    
    ## To redo files that havent be eaxracted as frames
    repeat_list =[]
    for roi_vid in processed_roi:
        folder_name = roi_vid.split('.')[0]
        if folder_name not in comp_processed_frames:
            repeat_list.append(roi_vid)
    
    print ("{} ROI extracted videos exist".format(len(processed_roi)))
    print("{} ND videos exist".format(len(processed_nd)))
    print("{} Videos with frames extraction incomplete, will be redone.".format(len(incomp_processed_frames)))
    print("{} videos not extracted as frames, will be redone".format(len(repeat_list)))
    
    ## Track Face and Extract Roi for all videos 
    print("In Progress: Roi Extraction.")
    data_folders = os.listdir(data_path)

    for folder in tqdm(data_folders):
        video_list = os.listdir(os.path.join(data_path,folder))
       
        for video_name in video_list :
            vidframe_folder = video_name.split('.')[0]
            
            if video_name in processed_roi :
                if video_name not in repeat_list:    
                    if vidframe_folder not in incomp_processed_frames :
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
    print("In Progress: Downsampling and Preparing labels/trial")
    
    label_files = os.listdir(label_path)
    ## Since we are using a stepsize of 5 for LSTM, we need to downsample singal to 128Hz to 10 Hz
    up,down = 25 , 64 
    for labels in tqdm(label_files):
        if labels.split('.')[-1] == 'mat' and len(labels.split(' '))==1 :
            #print(labels)
            labels_source = os.path.join(label_path,labels)
            y = f.loadLabels(labels_source)
            
            for i in range(len(y)):
                resampled = f.matchIoSr(y[i],up,down)
                resampled =np.array(resampled)
                derivative = f.getDerivative(resampled)
                
                save_name = labels.split('_')[0] + '_trial'+str(i+1)+'.dat' 
                save_path = os.path.join(labels_save_path,save_name)
                f.saveData(save_path,derivative)
    print("All labels saved as individual signals corresponding to videos")
    
    #f.plotHR(y[-1],128)
    #f.plotHR(resampled,`0)
    #t = f.loadData(save_path)
    #f.plotHR(t,10)

    ## Final Check to ensure every video has frames extracted ##
    redo_videos = []
    for video in tqdm(processed_roi):       
        if video.split('.')[0] not in comp_processed_frames:
            print(video)
            redo_videos.append(video)
    
    for videos in tqdm(redo_videos) :
        video = os.path.join(data_path,folder,video_name)
        f.getRoi(video, rsz_dim, roi_save_path, dataset_save_path)
    
    for vid in tqdm(redo_videos):
        if vid not in os.listdir(nd_save_path):
            vid = os.path.join (roi_save_path.as_posix(), vid_name)
            f.getNormalizedDifference(vid ,nd_save_path,dataset_save_path_nd)
    