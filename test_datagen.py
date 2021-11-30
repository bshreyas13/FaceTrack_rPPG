# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 04:13:59 2021

@author: bshreyas
"""
import os
import numpy as np
import tensorflow as tf
from modules.videodatasethandler import VideoDatasetHandler
from modules.preprocessor import Preprocessor 
# load the backend
from keras import backend as K
import argparse as ap

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("-m","--model", required = True , help = "FaceTrack_rPPG or DeepPhys")
    parser.add_argument("-id","--in_data", required = True , help = "Video name of format sXX_trialXX.avi")
    parser.add_argument("-bs", "--batch_size", required = True , help = "Desired batch size")
    args = vars(parser.parse_args())
     
    vdh = VideoDatasetHandler()
    p = Preprocessor()
    try:
        AUTOTUNE = tf.data.AUTOTUNE     
    except:
        AUTOTUNE = tf.data.experimental.AUTOTUNE 
    data = args['in_data']
    in_data = [data]
    model = args['model']
    batch_size = int(args['batch_size'])
    labels_path =  (os.path.join(os.path.dirname(os.getcwd()),'Labels'))
    appearance = os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Roi')
    motion = os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Nd') 
    datagen = vdh.dataGenerator(model, in_data, appearance, motion, labels_path)
    if model == 'DeepPhys':
        x_shape = (215,300,3)
        y_shape = ()

    elif model == 'FaceTrack_rPPG':
        x_shape = (5,215,300,3)
        y_shape = (5,)
        
    dataset = tf.data.Dataset.from_generator(
        generator=datagen, 
        output_types=((np.float64,np.float64), (np.float64)), 
        output_shapes=((x_shape,x_shape), (y_shape)))
    print(dataset.element_spec)
    # prevent Tensorflow memory leakage
    K.clear_session()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # Prefetch some batches.
    dataset = dataset.prefetch(AUTOTUNE)
    for (x_l,x_r),(y) in dataset.take(1):    
        print('Appeareance Input Shape:',x_l.numpy().shape)
        print('Motion Input Shape',x_r.shape)
        print('Output',y.shape)