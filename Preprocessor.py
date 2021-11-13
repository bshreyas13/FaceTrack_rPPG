# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:19:03 2021

@author: bshreyas
"""


import cv2
import pathlib
import numpy as np
import mediapipe as mp
import time 
import os
import scipy.io as sio  
import heartpy as hp
from scipy import signal 
import argparse as ap
from tqdm import tqdm


###########################################################################
## Class Preprocessor has funtions for facetracking , extracting ROI and ## 
## Preprocessing labels to feed the Deep network                         ##
###########################################################################

class Preprocessor:     
    ## Function draws the keypoints on video ##
    def drawMesh(self, image, facial_landmarks):
        height,width, _ = image.shape
        ## Draw Facemesh
        for i in range(0, 468):
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
            mesh_image = cv2.circle(image, (x, y), 2, (100, 100, 0))
        return mesh_image
    
    ## Track face and get ROI from Full video 
    def getRoi(self, video, rsz_dim, roi_save_path, log, save_tracked = False):
        
        
        roi = np.zeros(rsz_dim)
        ## Face Mesh setup
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3,min_tracking_confidence=0.75)
        
        ## Capture setup
        cap = cv2.VideoCapture(video)
        frame_width = int(cap.get(3)) 
        frame_height = int(cap.get(4)) 
        size = (frame_width, frame_height)
        
        ## Paths setup
        path_f = pathlib.PurePath(video)
        filename = path_f.name 
        
        ## Video Writer
        roi_out = cv2.VideoWriter(roi_save_path.as_posix() + '/'+ filename.split('.')[0] +'.avi',  
                                         cv2.VideoWriter_fourcc(*'MJPG'), 
                                         50, rsz_dim)


        if save_tracked == True:
            output = cv2.VideoWriter(filename.split('.')[0] + '_ft' +'.mp4',  
                                         cv2.VideoWriter_fourcc(*'MP4V'), 
                                         50, size) 
        ## Extract Frames
        while True:
            # Image
            ret, image = cap.read()
            if ret is not True:
                break
            height, width, _ = image.shape            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            ## Facial landmarks
            result = face_mesh.process(rgb_image)
            
            if result.multi_face_landmarks == None:
                roi_out.release()
                os.remove(roi_save_path.as_posix() + '/'+ filename.split('.')[0] +'.avi')
                log.append(filename)
                with open('tracking_fail_log.txt', 'a') as f:
                    for item in log:
                        f.write("%s\n" % item)
                break
            for facial_landmarks in result.multi_face_landmarks:
                
                ## Estimate Face bounding box from keypoints
                h, w, c = image.shape
                cx_min=  w
                cy_min = h
                cx_max= cy_max= 0
                for i, lm in enumerate(facial_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if cx<cx_min:
                        cx_min=cx
                    if cy<cy_min:
                        cy_min=cy
                    if cx>cx_max:
                        cx_max=cx
                    if cy>cy_max:
                        cy_max=cy
                ## Adjusting BB to include parts of forehead and neck
                cy_min = cy_min-25
                cy_max = cy_max+25
                
                ## get ROI from the obtained BB               
                roi = image[cy_min:cy_max,cx_min:cx_max,:]
                roi = cv2.resize(roi,rsz_dim)
                roi_out.write(roi)
                #print(roi.shape)
                
                ## condition to save tracking data video ##
                if save_tracked == True:
                    img = image.copy()
                    mesh_image = self.drawMesh(img, facial_landmarks,)
                    bb_image = cv2.rectangle(mesh_image, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)
            
            if save_tracked == True :
                output.write(bb_image)
        
        ## Close all open streams
        if save_tracked == True:
            output.release()
        roi_out.release()
        cap.release()
        cv2.destroyAllWindows()
                
        return roi
    
    ## Function to get Normalized difference frames for whole video ##
    ## uses the simplied idea of c'(t) = {c(t+1)-c(t)}/{c(t+1)+c(t)}##
    def getNormalizedDifference(self,video,nd_save_path):
        ## Capture setup 
        cap = cv2.VideoCapture(video)
        frame_width = int(cap.get(3)) 
        frame_height = int(cap.get(4)) 
        size = (frame_width, frame_height)
        
        ## Paths setup
        source_path = pathlib.PurePath(video)
        filename = source_path.name  
        
        ## Video writer
        output = cv2.VideoWriter(nd_save_path.as_posix() + '/'+ filename.split('.')[0] +'.avi',  
                                         cv2.VideoWriter_fourcc(*'MJPG'), 
                                         50, size) 
        frame_count = 0
        while True:
        
            ret, image = cap.read()
            if ret is not True:
                break
            height, width, _ = image.shape            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            frame_count += 1
            norm_diff = 0
            
            ## Address first frame
            if frame_count < 2 :
                frame = rgb_image.copy()
                c = frame_count
                norm_diff = np.zeros(rgb_image.shape)
                print(norm_diff.shape)
                output.write(norm_diff)
                #print(frame_count)
                continue
            
            ## All following frames 
            elif frame_count == c+1 :             
                frame_next = rgb_image.copy()
                c = frame_count  
                norm_diff = (frame_next - frame)/ (frame_next + frame)
                output.write(norm_diff)
                frame = rgb_image.copy()
                #print(frame_count)
        output.release()
        cap.release()
        
        return norm_diff
    
    ## Load .mat vectors for the ECG signal and trim the first three seconds
    ## rerturns a 40 x 7680 array of signals corresponding to 40 trials 
    def loadLabels(self, label_source):
        mat = sio.loadmat(label_source)
        sig_full = mat['dataECG']
        sig_trimmed = sig_full[:,128*3:]
        return sig_trimmed

    ## Plot the signal and heart rate obtained in BPM
    def plotHR(self,signal, sampling_rate):
        working_data, measures = hp.process(signal, sampling_rate)
        hp.plotter(working_data, measures)

    ## resample 128hz ECG signal to match 50fps rate of input video stream
    def matchIoSr(self,sig):
        resampled_sig = signal.resample_poly(sig,25,64)
        return resampled_sig
    
    ##Obtain first deravative of the signal 
    def getDerivative(self,sig):
        derivative = []
        count = 0
        for i in range (len(sig)):
            if i == 0 :
                x = sig[i].copy()
                count+=1
                derivative.append(0)
                continue
            elif count == i:
                x_next = sig[i].copy()
                derivative.append(x_next - x)
                x = sig[i].copy()
                count+=1
        return np.array (derivative)
    
    ## Function to save .dat file 
    def saveData(self,path ,sig):
        np.savetxt(path,sig)
        
    ## Function to load .dat file
    def loadData(self,path):
        data = np.genfromtxt(path)  
        return data
        
        
        
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
    
    #Check progress
    log = ['These videos failed face tracking']
    processed = os.listdir(roi_save_path)
    
    ## First Track Face and Extract Roi for all videos 
    print("Strating Roi Extraction.")
    data_folders = os.listdir(data_path)
  
    for folder in tqdm(data_folders):
        video_list = os.listdir(os.path.join(data_path,folder))
       
        for video_name in video_list :
            if video_name in processed :
                continue
            video = os.path.join(data_path,folder,video_name)
            img = f.getRoi(video, rsz_dim,roi_save_path,log)
    
    ## Get normalized difference frame  
    roi_vids = os.listdir(roi_save_path.as_posix())
    for vid_name in tqdm(roi_vids):
        vid = os.path.join (roi_save_path.as_posix(), vid_name)
        n_d = f.getNormalizedDifference( vid ,nd_save_path)
    
    end = time.time()
    print("All videos processed. Roi and Difference frames saved")
   
    #cv2.imwrite('test.png',img)
    #cv2.imwrite('test_nd.png',n_d)
    
    ## Process Labels (PPG signals)
    print("Downsampling and Preparing labels/trial")
    
    label_files = os.listdir(label_path)
    for labels in tqdm(label_files):
        if labels.split('.')[-1] == 'mat' and len(labels.split(' '))==1  :
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


    

