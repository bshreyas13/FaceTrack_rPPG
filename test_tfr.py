#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:24:53 2021

@author: bshreyas
"""

import os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse as ap

if __name__ == '__main__':
    
    parser = ap.ArgumentParser()
    parser.add_argument("-m","--model", required = True , help = "FaceTrack_rPPG or DeepPhys")
    parser.add_argument("-id","--in_data", required = True , help = "Video name of format sXX_trialXX.avi")
    parser.add_argument("-bs", "--batch_size", required = True , help = "Desired batch size")

    args = vars(parser.parse_args())
    
    print("Testing TFRecord")
    model =  args["model"]
    in_data = (args['in_data'])
    in_data = in_data.split(',')
    
    roi_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset','Roi'))               
    nd_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset','Nd'))
    labels_path =  pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Labels'))
        
    txt_files_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' ,'Example', 'Txt'))
    txt_files_path.mkdir(parents=True,exist_ok=True)
    
    tfrecord_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' ,'Example', 'TFRecords'))
    tfrecord_path.mkdir(parents=True,exist_ok=True)
    
    for label_file in os.listdir(labels_path):
        trial_num = label_file.split('trial')[-1].split('.')[0]
        if len(trial_num)==1 :
            new_name = label_file.split('trial')[0]+'trial'+'0'+trial_num+'.dat'
            # print(new_name)
            os.rename(os.path.join(labels_path,label_file),os.path.join(labels_path,new_name))
    
    if model == "FaceTrack_rPPG":
        from modules.tfrecordhandler import TFRWriter
        from modules.tfrecordhandler import TFRReader
    elif model == "DeepPhys" :
        from modules.tfrecordhandler_m2 import TFRWriter
        from modules.tfrecordhandler_m2 import TFRReader
    tfwrite = TFRWriter()
    roi = tfwrite.makeFiletxt(roi_path,nd_path, in_data,labels_path,txt_files_path) ## roi and nd together
    # txt = os.listdir(txt_files_path)
    
    file_list= tfwrite.makeShuffledDict(txt_files_path)

    ## Batching is done on Video level ==> use small batch size
    batch_size = int(args['batch_size'])
    
    # if batch_size > len(in_data):
    #     batch_size = len(in_data)
    timesteps=1
    split = "train"
    img_size = (215,300,3)

    try:
        AUTOTUNE = tf.data.AUTOTUNE     
    except:
        AUTOTUNE = tf.data.experimental.AUTOTUNE 

    tfwrite.writeTFRecords(roi_path,nd_path, txt_files_path, tfrecord_path, file_list, batch_size,'example',timesteps )
    tfrpath = tfrecord_path
    # make a dataset iterator
    data = TFRReader(batch_size, timesteps)
    batch = data.getBatch(tfrpath, True, 0)
    
    if model == 'FaceTrack_rPPG':
        ## Display samples 
        for (x_l,x_r),y, name, frame in batch.take(3):    
            print('Appeareance Input Shape:',x_r.shape)      
            print('Motion Input Shape',x_l.shape)
            print('Output',y.shape)
            print('Video name:',name.numpy())
        
        
            frames = frame.numpy().astype(str)[0].split('f')
            frames= [int(num) for num in frames if num]

            fig = plt.figure(figsize=(12,10))
    
            idx = 1
            # n_rows = batch_size*2
            n_rows = 4*2
        
            for i in range(0, batch_size):
            
                print('Displaying Video {}'.format(name.numpy()[i]))
                print('Displaying frames {}'.format(frames))
            
                for j in range(0, timesteps):    
                    ax = fig.add_subplot(n_rows, timesteps, idx)
                    ax.set_title(" frame {}".format(frames[j]))
                    ax.imshow(x_l.numpy()[i, j, : ,: ,:])
                    idx += 1
           
                for k in range(0,timesteps):
                    ax = fig.add_subplot(n_rows, timesteps, idx)
                    ax.set_title("frame {}".format(frames[k]))
                    ax.imshow(x_r.numpy()[i, k, : ,: ,:])
                    idx+=1
                
        plt.savefig('../Sample_inputs_{}.jpg'.format(model))
    
    else :
        
        ## Display samples 
        for (x_l,x_r),y, name, frame in batch.take(3):    
            print('Appeareance Input Shape:',x_r.shape)      
            print('Motion Input Shape',x_l.shape)
            print('Output',y.shape)
            print('Video name:',name.numpy())
        
        
            frames = frame.numpy().astype(str)[0].split('f')
            frames= [int(num) for num in frames if num]

            fig = plt.figure(figsize=(12,10))
    
            idx = 1
            # n_rows = batch_size*2
            n_rows = 4*2
        
            for i in range(0, batch_size):
            
                print('Displaying Video {}'.format(name.numpy()[i]))
                print('Displaying frames {}'.format(frames))
            
                for j in range(0, timesteps):    
                    ax = fig.add_subplot(n_rows, timesteps, idx)
                    ax.set_title(" frame {}".format(frames[j]))
                    ax.imshow(x_l.numpy()[i, j, : ,: ,:])
                    idx += 1
           
                for k in range(0,timesteps):
                    ax = fig.add_subplot(n_rows, timesteps, idx)
                    ax.set_title("frame {}".format(frames[k]))
                    ax.imshow(x_r.numpy()[i, k, : ,: ,:])
                    idx+=1
                
        plt.savefig('../Sample_inputs_{}.jpg'.format(model))
