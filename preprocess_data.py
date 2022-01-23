# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:06:53 2021

@author: bshreyas
"""
import argparse as ap
from tqdm import tqdm
import time 
import os
import pathlib
import numpy as np
import sys
from modules.preprocessor import Preprocessor
from modules.videodatasethandler import VideoDatasetHandler

    #####################################################################
    ## Script to process all videos in the Deap folder                 ##
    ## Each subject has 40 trials of 1 minute each                     ##
    ## There is one label file with 40 signals corresponding to trials ##
    #####################################################################
    
if __name__ == '__main__':
    
    parser = ap.ArgumentParser()
    parser.add_argument("-ds","--data_source", required = True , help = "path to video source directory with unsplit dataset")
    parser.add_argument("-lp","--label_path", required = True , help = "path to directory with labels .mat signals")
    parser.add_argument("-fc","--final_check", action ='store_true', required = False , help = "Toggle to enable only final intergrity check")
    parser.add_argument("-fo","--frames_only", action ='store_true', required = False , help = "Toggle to enable extract frames only, for use when you already have Preproessed videos")
    parser.add_argument("-nl","--normalize_labels", action ='store_true', required = False , help = "Toggle to enable Label normalization")
    parser.add_argument("-nc","--no_crop", action ='store_true', required = False , help = "Toggle to process DEAP without Cropping")
    parser.add_argument("-sbst","--subset", required = False , help = "Subset of videos to process in each subject")
    parser.add_argument("-ims", "--image_size", required = False , help = "Desired input img size.")
    args = vars(parser.parse_args())
    
    data_path = args['data_source']
    label_path = args['label_path']
    fc = args['final_check']
    fo = args['frames_only']
    nl = args['normalize_labels']
    nc = args["no_crop"]
    subset = float(args["subset"])
    img_size = args("image_size")
    if img_size == None:
            img_size = "215X300X3"
        else:
            img_size = args["image_size"]
    ## Intialize preprocessor 
    f = Preprocessor()
    vdh = VideoDatasetHandler()
    
    ## Get Roi for all videos ##
    
    ## Resize roi videos to standardize dims 
    rsz_dim = (300,215)
    
    roi_save_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Roi_Videos'))           
    roi_save_path.mkdir(parents=True,exist_ok=True)
    
    nd_save_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'ND_Videos'))
    nd_save_path.mkdir(parents=True,exist_ok=True)
    
    labels_save_path =  pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Labels'))
    labels_save_path.mkdir(parents=True,exist_ok=True)
    
    labels_save_path_ =  pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Scaled_labels'))
    labels_save_path_.mkdir(parents=True,exist_ok=True)

    dataset_save_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Roi'))
    dataset_save_path.mkdir(parents=True,exist_ok=True)

    dataset_save_path_nd = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Nd'))
    dataset_save_path_nd.mkdir(parents=True,exist_ok=True)
    

    ## Use -nc flag to process original videos without face tracking ##
    if nc == True:
        
        
        nd_save_path_og = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'ND_Videos_Nc'))
        nd_save_path_og.mkdir(parents=True,exist_ok=True)
        
        dataset_save_path_og = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset_OG' , 'RGB'))
        dataset_save_path_og.mkdir(parents=True,exist_ok=True)

        dataset_save_path_og_nd = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset_OG' , 'Nd'))
        dataset_save_path_og_nd.mkdir(parents=True,exist_ok=True)

        processed_og = os.listdir(dataset_save_path_og.as_posix())
        repeat_list =[]

        ## To redo files that havent be eaxracted as frames
        incomp_processed_frames, comp_processed_frames = vdh.verifyDataset(dataset_save_path_og)
        if len(processed_og) != len(comp_processed_frames) or len(incomp_processed_frames) != 0:            
            for vid in processed_og:
                folder_name = vid.split('.')[0]
                if folder_name not in comp_processed_frames:
                    repeat_list.append(vid)
            print("{} Videos with frames extraction incomplete, will be redone.".format(len(incomp_processed_frames)))
            print("{} videos not extracted as frames, will be redone".format(len(repeat_list)))

        
        print("In Progress: producing ND stream and Frame extraction without cropping roi")
        print("Using {}% videos for each subject".format(subset*100))
        ## Get normalized difference frame  
        
        data_folders = os.listdir(data_path)
        subjects = []
        for folder in data_folders:
            if not folder.startswith('.') :
                if not folder.startswith('_'):
                    subjects.append(folder)

        for folder in tqdm(subjects):
            video_list = os.listdir(os.path.join(data_path,folder))
            video_list = vdh.getSubset(video_list,subset)
            for video_name in video_list :
                
                vidframe_folder = video_name.split('.')[0]
                if video_name in processed_og:
                    if video_name not in repeat_list:    
                        if vidframe_folder not in incomp_processed_frames :
                            continue

                video = os.path.join(data_path,folder,video_name)
                with open('log_processed_nc.txt', 'a') as file:
                            file.write("%s\n" %video_name )
                #n_d = f.getNormalizedDifference( video ,nd_save_path_og,dataset_save_path_og_nd)
                f.resizeAndGetND_V(video ,dataset_save_path_og,dataset_save_path_og_nd,img_size)
                
        sys.exit()

    ## Use flag to only standardize labels
    if nl == True:
        try:
            for label_file in os.listdir(labels_save_path):
                label = os.path.join(labels_save_path,label_file)
                label_scaled = f.normalizeLabels(label) 
                save_path = os.path.join(labels_save_path_,label_file)
                f.saveData(save_path,label_scaled)
            print("Scaled label files saved")
            sys.exit()
        except:
            print("check if label files per video exist at given path")
    
    ## Check and display progress
    processed_roi = os.listdir(roi_save_path)
    processed_nd = os.listdir(nd_save_path)    
    repeat_list =[]
    
    ## Flag for frame extraction only from ND and Roi videos 
    if fo == True:
        try:
            print("In Progress: Frame extraction")
            for video_name in tqdm(processed_roi):
                if video_name in processed_nd:
                    vid_roi = os.path.join(roi_save_path.as_posix(),video_name)
                    vid_nd = os.path.join(nd_save_path.as_posix(),video_name)
                    f.getFramesOnly(vid_roi ,dataset_save_path)
                    f.getFramesOnly(vid_nd , dataset_save_path_nd)
            
            print("All frame extracted")
            incomp_processed_frames, comp_processed_frames = vdh.verifyDataset(dataset_save_path)
            incomp_processed_ndf,comp_processed_ndf = vdh.verifyDataset(dataset_save_path_nd)
            redo_videos = []
            for video in tqdm(processed_roi):       
                if video.split('.')[0] not in comp_processed_frames:
                    print(video)
                    redo_videos.append(video)
        
            redo_videos_nd = []
            for video in tqdm(processed_nd):       
                if video.split('.')[0] not in comp_processed_ndf:
                    print(video)
                    redo_videos_nd.append(video)
        except:
            print("No Preprocessed Videos tun off -fo")
    
    ## Flag for Data integrity check after processing         
    elif fc == True:
        ## Final Check to ensure every video frames are extracted ##
        incomp_processed_frames, comp_processed_frames = vdh.verifyDataset(dataset_save_path)
        incomp_processed_ndf,comp_processed_ndf = vdh.verifyDataset(dataset_save_path_nd)
        
        redo_videos=[]
        for video in tqdm(processed_roi):       
            if video.split('.')[0] not in comp_processed_frames:
                print(video)
                redo_videos.append(video)
        
        redo_videos_nd = []
        for video in tqdm(processed_nd):       
            if video.split('.')[0] not in comp_processed_ndf:
                print(video)
                redo_videos_nd.append(video)
        print("{} Videos with roi frames extraction incomplete".format(len(incomp_processed_frames)))
        print("{} videos not extracted as frames".format(len(redo_videos)))       
        print("{} Videos with ND frames extraction incomplete, will be redone.".format(len(incomp_processed_ndf)))
        print("{} videos not extracted as frames, will be redone".format(len(redo_videos_nd)))

    
    ## Original Run with roi extraction , ND and label processing 
    else:       
        ## To redo files that havent be eaxracted as frames
        incomp_processed_frames, comp_processed_frames = vdh.verifyDataset(dataset_save_path)
        if len(processed_roi) != len(comp_processed_frames) or len(incomp_processed_frames) != 0:            
            for roi_vid in processed_roi:
                folder_name = roi_vid.split('.')[0]
                if folder_name not in comp_processed_frames:
                    repeat_list.append(roi_vid)
            print("{} Videos with frames extraction incomplete, will be redone.".format(len(incomp_processed_frames)))
            print("{} videos not extracted as frames, will be redone".format(len(repeat_list)))

        
        with open('log_fail.txt') as skip:
            skip_list = skip.readlines()
    
        print ("{} ROI extracted videos exist".format(len(processed_roi)))
        print("{} ND videos exist".format(len(processed_nd)))

    
        ## Track Face and Extract Roi for all videos 
        print("In Progress: Roi Extraction.")
        data_folders = os.listdir(data_path)
        
        for folder in tqdm(data_folders):
            video_list = os.listdir(os.path.join(data_path,folder))
            
            for video_name in video_list :
                vidframe_folder = video_name.split('.')[0]
            
                if video_name in processed_roi:
                    if video_name not in repeat_list:    
                        if vidframe_folder not in incomp_processed_frames :
                            continue
             
                elif video_name in skip_list:
                    continue
            
                video = os.path.join(data_path,folder,video_name)
                with open('log_processed.txt', 'a') as file:
                            file.write("%s\n" %video_name )
                img = f.getRoi(video, rsz_dim, roi_save_path, dataset_save_path)
        print("Roi Extraction complete\n")    
    
        print("In Progress: Normalized Difference stream extraction")
        ## Check for previously extracted data
        repeat_list =[]
        incomp_processed_ndf,comp_processed_ndf = vdh.verifyDataset(dataset_save_path_nd)
        if len(processed_nd) != len(os.listdir(dataset_save_path_nd)) or len(incomp_processed_ndf)!= 0:      
            for roi_vid in processed_roi:
                folder_name = roi_vid.split('.')[0]
                if folder_name not in comp_processed_frames:
                    repeat_list.append(roi_vid)
            print("{} Videos with ND frames extraction incomplete, will be redone.".format(len(incomp_processed_ndf)))
            print("{} videos not extracted as frames, will be redone".format(len(repeat_list)))
        
        with open('ND_issues.txt') as bk:
            backup = bk.readlines()
        
        ## Get normalized difference frame  
        roi_vids = os.listdir(roi_save_path.as_posix())
        for vid_name in tqdm(roi_vids):
            if vid_name not in processed_nd:
                if vid_name not in repeat_list:
                   if vid_name.split('.')[0] in incomp_processed_ndf :
                        continue
                vid = os.path.join (roi_save_path.as_posix(), vid_name)
                n_d = f.getNormalizedDifference( vid ,nd_save_path,dataset_save_path_nd)
            elif vid_name in backup :
                vid = os.path.join (roi_save_path.as_posix(), vid_name)
                n_d = f.getNormalizedDifference( vid ,nd_save_path,dataset_save_path_nd)
        print("All videos processed. Roi and Difference frames saved")
    
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
    
   
    