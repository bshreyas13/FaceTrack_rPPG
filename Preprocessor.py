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
import matplotlib.pyplot as plt
import heartpy as hp
from scipy import signal 

###########################################################################
## Class Preprocessor has funtions for facetracking , extracting ROI and ## 
## Preprocessing labels to feed the Deep network                         ##
###########################################################################

class Preprocessor:
    
    def __init__(self,data_source, label_source):
        self.data_source = data_source
        self.label_source = label_source
        
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
    def getRoi(self, rsz_dim, roi_save_path, save_tracked = False):
        
        ## Face Mesh setup
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()
        
        ## Capture setup
        cap = cv2.VideoCapture(self.data_source)
        frame_width = int(cap.get(3)) 
        frame_height = int(cap.get(4)) 
        size = (frame_width, frame_height)
        
        ## Paths setup
        path_f = pathlib.PurePath(self.data_source)
        filename = path_f.name 
        roi_save_path = pathlib.Path(roi_save_path)
        roi_save_path.mkdir(parents=True,exist_ok=True)
        
        ## Video Writer
        roi_out = cv2.VideoWriter(roi_save_path.as_posix() + '/'+ filename.split('.')[0] +'_roi'+'.mp4',  
                                         cv2.VideoWriter_fourcc(*'MP4V'), 
                                         50, rsz_dim)


        if save_tracked == True:
            output = cv2.VideoWriter(filename.split('.')[0] +'.mp4',  
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
        nd_save_path = pathlib.Path(nd_save_path)
        nd_save_path.mkdir(parents=True,exist_ok=True)
        
        ## Video writer
        output = cv2.VideoWriter(nd_save_path.as_posix() + '/'+ filename.split('.')[0]+'_n_diff' +'.mp4',  
                                         cv2.VideoWriter_fourcc(*'MP4V'), 
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
        return norm_diff
    
    ## Load .mat vectors for the ECG signal and trim the first three seconds
    ## rerturns a 40 x 7680 array of signals corresponding to 40 trials 
    def load_labels(self):
        mat = sio.loadmat(self.label_source)
        sig_full = mat['dataECG']
        sig_trimmed = sig_full[:,128*3:]
        return sig_trimmed

    ## Plot the signal and heart rate obtained in BPM
    def plot_HR(self,signal, sampling_rate):
        working_data, measures = hp.process(signal, sampling_rate)
        hp.plotter(working_data, measures)

    ## resample 128hz ECG signal to match 50fps rate of input video stream
    def match_io_sr(self,sig):
        resampled_sig = signal.resample_poly(sig,25,64)
        return resampled_sig

       
        output.release()
        cap.release()
        
        return norm_diff
        
            
if __name__ == '__main__':
    
    f = Preprocessor('s01_trial16.avi','S01_ECG.mat')
    start = time.time()
    rsz_dim = (300,215)
    roi_save_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Roi_Videos'))
    nd_save_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'ND_Videos'))
    r = roi_save_path.as_posix()
    img = f.getRoi(rsz_dim,roi_save_path)
    n_d = f.getNormalizedDifference( roi_save_path.as_posix() + '/s01_trial16_roi.mp4',nd_save_path)
    end = time.time()
    print(end-start)
    cv2.imwrite('test.png',img)
    cv2.imwrite('test_nd.png',n_d)
    
    x = f.load_labels()
    f.plot_HR(x[0],128)
    r = f.match_io_sr(x[0])
    f.plot_HR(r,50)
