# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:19:03 2021

@author: bshreyas
"""


import cv2
import pathlib
import numpy as np
import mediapipe as mp
import os
import shutil
import scipy.io as sio  
import heartpy as hp
from scipy import signal 
from sklearn.preprocessing import StandardScaler

###########################################################################
## Class Preprocessor has funtions for facetracking , extracting ROI and ## 
## Preprocessing labels to feed the Deep network                         ##
###########################################################################

class Preprocessor:     
    
    ## Function draws the keypoints on video ##
    ## Returns a np array image with face mest drawn on top ##
    def drawMesh(self, image, facial_landmarks):
        height,width, _ = image.shape
        ## Draw Facemesh
        for i in range(0, 468):
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
            mesh_image = cv2.circle(image, (x, y), 2, (100, 100, 0))
        return mesh_image
    
    ## Function Track face and get ROI from Full video ##
    ## Returns the ROI of last frame of the video as an np.array ##
    ## Saves roi extracted videos and roi extracted frames as folders of .jpg ## 
    def getRoi(self, video, rsz_dim, roi_save_path, dataset_save_path, save_tracked = False):
        
        
        roi = np.zeros(rsz_dim)
        ## Face Mesh setup
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)
        
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
        frame_count =0 
        while True:
            # Image
            ret, image = cap.read()
            if ret is not True:
                break
            height, width, _ = image.shape            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_count +=1
            ## Facial landmarks
            result = face_mesh.process(rgb_image)
            
            if result.multi_face_landmarks == None:
                roi_out.release()
                os.remove(roi_save_path.as_posix() + '/'+ filename.split('.')[0] +'.avi')
                frames_save_path =  pathlib.Path(os.path.join(dataset_save_path,filename.split('.')[0]))
                if os.path.isdir(frames_save_path):
                    shutil.rmtree(os.path.join(dataset_save_path,filename.split('.')[0]))
                with open('tracking_fail_log.txt', 'a') as f:
                    f.write("%s\n" % filename)
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
                self.saveFrames(roi,dataset_save_path,filename,frame_count)
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
    
    ## Function to get Normalized difference(Dirivative) frames for whole video ##
    ## uses the simplied idea of c'(t) = {c(t+1)-c(t)}/{c(t+1)+c(t)}##
    ## Returns the ND frame of last frame of the video as an np.array ##
    ## Saves ND extracted videos and ND frames as folders of .jpg ## 
    def getNormalizedDifference(self,video,nd_save_path,dataset_save_path_nd):
        ## Capture setup 
        cap = cv2.VideoCapture(video)
        frame_width = int(cap.get(3)) 
        frame_height = int(cap.get(4)) 
        size = (frame_width, frame_height)
        
        ## Paths setup
        source_path = pathlib.PurePath(video)
        filename = source_path.name  
        
        ## Video writer
        #output = cv2.VideoWriter(nd_save_path.as_posix() + '/'+ filename.split('.')[0] +'.avi', 
         #                                cv2.VideoWriter_fourcc(*'MJPG'), 
          #                               50, size) 
        frame_count = 0
        norm_diff = 0
        while True:
            
            ret, image = cap.read()
            if ret is not True:
                break
            height, width, _ = image.shape            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            frame_count += 1
            
            ## Address first frame
            if frame_count < 2 :
                frame = rgb_image.copy()
                c = frame_count
                norm_diff = np.zeros(rgb_image.shape)
                norm_diff = np.uint8(255*norm_diff)
                self.saveFrames(norm_diff,dataset_save_path_nd,filename,frame_count)
                #output.write(norm_diff)
                #print(frame_count)
                continue
            
            ## All following frames 
            elif frame_count == c+1 :             
                frame_next = rgb_image.copy()
                c = frame_count  
                norm_diff = (frame_next - frame)/ (frame_next + frame)
                norm_diff = np.uint8(255*norm_diff)
                self.saveFrames(norm_diff,dataset_save_path_nd,filename,frame_count)                
                #output.write(norm_diff)
                frame = rgb_image.copy()
                #print(frame_count)
            else :
                with open('log_ND_issues.txt', 'a') as f:
                    f.write("%s\n" % filename)
        #output.release()
        cap.release()
        
        return norm_diff
    
    ## Function to get Normalized difference(Dirivative) frames for whole video ##
    ## uses the simplied idea of c'(t) = {c(t+1)-c(t)}/{c(t+1)+c(t)}##
    ## Returns the ND frame of last frame of the video as an np.array ##
    ## Saves ND extracted videos and ND frames as folders of .jpg ## 
    def resizeAndGetND_V(self,video,dataset_save_path_rgb, dataset_save_path_nd,img_size):
        ## Capture setup 
        cap = cv2.VideoCapture(video)
        frame_width = int(cap.get(3)) 
        frame_height = int(cap.get(4)) 
        size = (frame_width, frame_height)
        
        ## Paths setup
        source_path = pathlib.PurePath(video)
        filename = source_path.name  
        
        frame_count = 0
        
        while True:
            
            ret, image = cap.read()
            if ret is not True:
                break
            height, width, _ = image.shape            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            w , h = 496,496
            center = [rgb_image.shape[0] / 2, rgb_image.shape[1]/2]
            x = center[1] - w/2
            y = center[0] - h/2
            crop_img = rgb_image[int(y):int(y+h), int(x):int(x+w)]         
            ## Resize for spatial averaging
            size = (img_size[0],img_size[1])
            rgb_image = cv2.resize(crop_img,size,interpolation = cv2.INTER_CUBIC)
            
            frame_count += 1
            
            ## Address first frame
            if frame_count < 2 :
                frame = rgb_image.copy()
                c = frame_count
                norm_diff = np.zeros(rgb_image.shape)
                norm_diff = np.uint8(255*norm_diff)
                self.saveFrames(rgb_image,dataset_save_path_rgb,img_name,frame_count)
                self.saveFrames(norm_diff,dataset_save_path_nd,filename,frame_count)
                
                #print(frame_count)
                continue
            
            ## All following frames 
            elif frame_count == c+1 :             
                frame_next = rgb_image.copy()
                c = frame_count  
                norm_diff = (frame_next - frame)/ (frame_next + frame)
                norm_diff = np.uint8(255*norm_diff)
                self.saveFrames(rgb_image,dataset_save_path_rgb,img_name,frame_count)
                self.saveFrames(norm_diff,dataset_save_path_nd,filename,frame_count)                
                
                frame = rgb_image.copy()
                #print(frame_count)
            else :
                with open('log_ND_issues.txt', 'a') as f:
                    f.write("%s\n" % filename)
        #output.release()
        cap.release()

    ## TO get spatial averaged data from already frrame extacrted datatset
    ## in_path : path to video frames
    ## vid_fodler : the folder being processed
    ## data_save_path_nd : save path for output
    def resizeAndGetND(self,in_path, vid_folder,dataset_save_path_nd, img_size):
        
        frame_count = 1
        ## Address first frame
        img_list = os.listdir(video_folder)
        for img_name in img_list:
            rgb_image = cv.imread(os.path.join(in_path,vid_folder,img_name))
            ## Center Crop
            w , h = 496,496
            center = center = [rgb_image.shape[0] / 2, rgb_image.shape[1]/2]
            x = center[1] - w/2
            y = center[0] - h/2
            crop_img = rgb_image[int(y):int(y+h), int(x):int(x+w)]         
            ## Resize for spatial averaging
            size = (img_size[0],img_size[1])
            rgb_image = cv2.resize(crop_img,size,interpolation = cv2.INTER_CUBIC)
            
            if frame_count < 2 :
                frame = rgb_image.copy()
                c = frame_count
                norm_diff = np.zeros(rgb_image.shape)
                norm_diff = np.uint8(255*norm_diff)
                self.saveFrames(norm_diff,dataset_save_path_nd,filename,frame_count)
                frame_count+=1
                #print(frame_count)
                continue
            
            ## All following frames 
            elif frame_count == c+1 :             
                frame_next = rgb_image.copy()
                c = frame_count  
                norm_diff = (frame_next - frame)/ (frame_next + frame)
                norm_diff = np.uint8(255*norm_diff)
                self.saveFrames(norm_diff,dataset_save_path_nd,filename,frame_count)                
                frame = rgb_image.copy()
                #print(frame_count)
                frame_count+=1
            
            rgb_save_path = os.path.join(in_path,vid_folder)
            self.saveFrames(rgb_image,rgb_save_path,img_name,frame_count)

    def getFramesOnly(self,video,dataset_save_path):
        ## Capture setup 
        cap = cv2.VideoCapture(video)
        
        ## Paths setup
        source_path = pathlib.PurePath(video)
        filename = source_path.name
        frame_count = 0
        while True:
            
            ret, image = cap.read()
            if ret is not True:
                break
            frame_count +=1
            height, width, _ = image.shape            
            #rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.saveFrames(image,dataset_save_path,filename,frame_count) 
        cap.release()
        
           
    ## Load .mat vectors for the ECG signal and trim the first three seconds ##
    ## rerturns a 40 x 7680 array of signals corresponding to 40 trials ##
    def loadLabels(self, label_source):
        mat = sio.loadmat(label_source)
        sig_full = mat['dataECG']
        sig_trimmed = sig_full[:,128*3:]
        return sig_trimmed

    ## Plot the signal and heart rate obtained in BPM
    def plotHR(self,signal, sampling_rate):
        working_data, measures = hp.process(signal, sampling_rate)
        hp.plotter(working_data, measures)

    ## Function to resample 128hz ECG signal to match 50fps rate of input video stream ##
    ## Returns the resmapled signal array ##
    def matchIoSr(self,sig,up,down):
        
        resampled_sig = signal.resample_poly(sig,up,down)
        return resampled_sig
    
    ## Funtion to obtain first deravative of the signal ##
    ## Return an array with deravative signal ##
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
    
    ## Function to create folder and write images
    def saveFrames(self,img, dataset_save_path,filename,frame_count):
        frames_save_path =  pathlib.Path(os.path.join(dataset_save_path,filename.split('.')[0]))
        frames_save_path.mkdir(parents=True,exist_ok=True)                
        cv2.imwrite(frames_save_path.as_posix() + '/{}'.format(filename.split('.')[0]) + '_f{}.jpg'.format(frame_count), img)
    
    def normalizeLabels(self,label_path):

        scaler = StandardScaler()
        labels = self.loadData(label_path)
        labels_scaled = scaler.fit_transform(labels.reshape(-1,1))

        return labels_scaled
