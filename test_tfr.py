#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:24:53 2021

@author: bshreyas
"""
## Update Notes ##

## Fix sample plotting for DeepPhys ##

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
    parser.add_argument("-ims", "--image_size", required = False , help = "Desired batch size")

    args = vars(parser.parse_args())
    
    print("Testing TFRecord")
    model =  args["model"]
    in_data = (args['in_data'])
    in_data = in_data.split(',')
    img_size = args["image_size"]
    if img_size == None:
            img_size = "215X300X3"
    else:
        img_size = args["image_size"]
        
    img_size = [int(dim) for dim in img_size.split('X')]
    img_size = (img_size[0],img_size[1],img_size[2])
    roi_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset_OG','Roi'))               
    nd_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset_OG','Nd'))
    labels_path =  pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Scaled_labels'))
        
    txt_files_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' ,'Example', 'Txt'))
    txt_files_path.mkdir(parents=True,exist_ok=True)
    
    tfrecord_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' ,'Example', 'TFRecords'))
    tfrecord_path.mkdir(parents=True,exist_ok=True)
    
    
    if model == "FaceTrack_rPPG":
        from modules.tfrecordhandler_FTR import TFRWriter
        from modules.tfrecordhandler_FTR import TFRReader
    elif model == "DeepPhys" :
        from modules.tfrecordhandler_DP import TFRWriter
        from modules.tfrecordhandler_DP import TFRReader
    elif model == "DeepPhys_V0" :
        from modules.tfrecordhandler_DP import TFRWriter
        from modules.tfrecordhandler_DP import TFRReader
    tfwrite = TFRWriter(img_size)
    roi = tfwrite.makeFiletxt(roi_path,nd_path, in_data,labels_path,txt_files_path) ## roi and nd together
    # txt = os.listdir(txt_files_path)
    
    file_list= tfwrite.makeShuffledDict(txt_files_path)

    ## Batching is done on Video level ==> use small batch size
    batch_size = int(args['batch_size'])
    
    # if batch_size > len(in_data):
    #     batch_size = len(in_data)
    timesteps=1
    split = "example"
    

    try:
        AUTOTUNE = tf.data.AUTOTUNE     
    except:
        AUTOTUNE = tf.data.experimental.AUTOTUNE 

    tfwrite.writeTFRecords(roi_path,nd_path, txt_files_path, tfrecord_path, file_list, batch_size,split,timesteps )
    tfrpath = os.path.join(tfrecord_path.as_posix(),split)
    # make a dataset iterator
    data = TFRReader(batch_size, timesteps)
    if model == 'FaceTrack_rPPG':
        batch = data.getBatch(tfrpath, 1, True, 0)
        print("Batch retrieved")

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
        
        batch = data.getBatch(tfrpath, 1, False, 1)
        print("Batch retrieved")

        for (x_l,x_r),(y) in batch.take(1):    
            print('Appeareance Input Shape:',x_r.shape)      
            print('Motion Input Shape',x_l.shape)
            print('Output',y.shape)
            print(y.numpy())
            fig = plt.figure(figsize=(12,10))

            idx = 1
            # n_rows = batch_size*2
            n_rows = 2
        
            for i in range(2):                   
                ax = fig.add_subplot(n_rows, 2, idx)
                ax.imshow(x_l.numpy()[i, : ,: ,:])
                ax = fig.add_subplot(n_rows, 2, idx+2)
                ax.imshow(x_r.numpy()[i, : ,: ,:])
                idx += 1
       
        plt.savefig('../Sample_inputs_{}.jpg'.format(model))
