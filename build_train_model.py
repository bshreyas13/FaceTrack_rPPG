# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:19:24 2021

@author: bshreyas
"""
import ArgumentParser as ap
from modules.models import Models
from tensorflow.keras.utils import plot_model

if __name__ == '__main__':
    
    parser = ap.ArgumentParser()
    parser.add_argument("-mn","--model_name", required = True , help = "Facetrack_rPPG or DeepPhys")
    parser.add_argument("-ts","--timesteps", required = False , help = "timestep for FaceTrack_rPPG, defaults to 5")
    args = vars(parser.parse_args())
    
    if args["timesteps"] == None:    
        timesteps = 5
    else:
        timesteps = args["timesteps"]
        
    input_shape = (timesteps,300,215,3)
    n_filters =32
    batch_size = 100

    model= Models.FaceTrack_rPPG(input_shape, timesteps, n_filters)

    # verify the model using graph

    plot_model(model, to_file='FaceTrack_rPPG.png', show_shapes=True)
    model.summary()