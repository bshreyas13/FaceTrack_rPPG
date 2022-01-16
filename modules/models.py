# -*- coding: utf-8 -*-
"""
Created on Monday Nov 15 02:10:17 2021

@author: bshreyas
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, AveragePooling3D,AveragePooling2D, ConvLSTM2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout

class Models:
    #################################################################
    ## The model here uses a Many to one LSTM approach             ##
    ## input shape : (samples, timestep, height, width, channels ) ##
    ## output shape : (samples,5)                                  ##
    #################################################################
    def FaceTrack_rPPG(input_shape, timesteps, n_filters, n_layers=2,kernel_size=(3,3)):
        
        filters = n_filters
        ## Left branch of Y network
        left_inputs = Input(shape=input_shape)
        x = left_inputs ## Appearance
        
        ## Right branch of Y network
        right_inputs = Input(shape=input_shape)
        y = right_inputs  ## Motion
        

        # 2 layers of ConvLSTM2D-AveragePooling3D/2D
        # number of filters doubles after each layer (32-64-128)
        for i in range(n_layers):
      
            if i == n_layers-1 :
                x = ConvLSTM2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               activation='tanh',
                               data_format = 'channels_last',
                               kernel_initializer='glorot_uniform',
                               return_sequences = False)(x)
                x = AveragePooling2D(pool_size=(1,2))(x)
            else:
                x = ConvLSTM2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               activation='tanh',
                               data_format = 'channels_last',
                               kernel_initializer='glorot_uniform',
                               return_sequences = True)(x)
                #x = Dropout(dropout)(x)
                x = AveragePooling3D(pool_size=(1,2,2))(x)
            if i == n_layers-1 :
                y = ConvLSTM2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               activation='tanh',
                               data_format = 'channels_last',
                               kernel_initializer='glorot_uniform',
                               return_sequences = False)(y)
                y = AveragePooling2D(pool_size=(1,2))(y)
            else:
                y = ConvLSTM2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               activation='tanh',
                               data_format = 'channels_last',
                               kernel_initializer='glorot_uniform',
                               return_sequences = True)(y)
                #y = Dropout(dropout)(y)
                y = AveragePooling3D(pool_size=(1,2,2))(y)
            filters *= 2
            y = tf.math.multiply(x,y, name ='Elementwise Multiplication')
        # Feature maps to vector before connecting to Dense 
        y = Flatten()(y)
        y = Dense(128,kernel_initializer='glorot_uniform')(y)
        outputs = Dense(timesteps)(y)
        # Build the model (functional API)
        model = Model([left_inputs, right_inputs], outputs,name = 'FaceTrack_rPPG')
        return model
    
    #####################################################################
    ## The  DeepPhys model is implemented as per paper                 ##
    ## input shape : (samples, height, width, channels )               ##
    ## output shape : (samples, 1)                                     ##
    #####################################################################
    def DeepPhys(input_shape, n_filters, n_layers=2, kernel_size=(3,3)):
        
        filters = n_filters
        ## Left branch of Y network
        left_inputs = Input(shape=input_shape)
        x = left_inputs ## Appearance
        
        ## Right branch of Y network
        right_inputs = Input(shape=input_shape)
        y = right_inputs  ## Motion
        
        
        # 2 layers of Conv2D-AvgPooling2D
        
        for i in range(n_layers):
            x = Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 activation='tanh')(x)
            #x = Dropout(dropout)(x)
            x = AveragePooling2D(pool_size=(2,2))(x)
            
            y = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       padding='same',
                       kernel_initializer='glorot_uniform',
                       activation='tanh')(y)
            #y = Dropout(dropout)(y)
            y = AveragePooling2D(pool_size=(2,2))(y)
            filters *= 2
            y = tf.math.multiply(x,y, name ='Elementwise Multiplication')

        # Feature maps to vector before connecting to Dense 
        y = Flatten()(y)
        y = Dense(128,kernel_initializer='glorot_uniform')(y)
        outputs = Dense(1)(y)
        # Build the model (functional API)
        model = Model([left_inputs, right_inputs], outputs, name='DeepPhys') 
        return model