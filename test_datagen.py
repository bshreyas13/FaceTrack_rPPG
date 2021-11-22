# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 04:13:59 2021

@author: local_test
"""
import os
import numpy as np
import tensorflow as tf
from modules.videodatasethandler import VideoDatasetHandler
from modules.preprocessor import Preprocessor 
import argparse as ap

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("-m","--model", required = True , help = "FaceTrack_rPPG or DeepPhys")
    parser.add_argument("-id","--in_data", required = True , help = "Video name of format sXX_trialXX.avi")
    parser.add_argument("-bs", "--batch_size", required = True , help = "Desired batch size")
    args = vars(parser.parse_args())
     
    vdh = VideoDatasetHandler()
    p = Preprocessor()

    in_data = args['in_data']
    model = args['model']
    batch_size = int(args['batch_size'])
    labels_path =  (os.path.join(os.path.dirname(os.getcwd()),'Labels'))
    roi = os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Roi')
    datagen = vdh.dataGenerator(model, in_data, roi ,labels_path)
    if model == 'DeepPhys':
        x_shape = (215,300,3)
        y_shape = ()

    elif model == 'FaceTrack_rPPG':
        x_shape = (5,215,300,3)
        y_shape = (5,)
        
    dataset = tf.data.Dataset.from_generator(
        generator=datagen, 
        output_types=(np.float64, np.float64), 
        output_shapes=(x_shape, y_shape))
    print(dataset.element_spec)
    for x,y in dataset.repeat().batch(10).take(1):
        print(x.numpy().shape)
        print(y.numpy().shape)