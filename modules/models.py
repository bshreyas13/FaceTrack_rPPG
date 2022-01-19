# -*- coding: utf-8 -*-
"""
Created on Monday Nov 15 02:10:17 2021

@author: bshreyas
"""
## Update Notes:
## Fix Attention Masks in deepPhys

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input,BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling3D,AveragePooling2D, ConvLSTM2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
import sys 

class Models:
    #################################################################
    ## The model here uses a Many to one LSTM approach             ##
    ## input shape : (samples, timestep, height, width, channels ) ##
    ## output shape : (samples,5)                                  ##
    #################################################################
    def FaceTrack_rPPG(input_shape, timesteps, n_filters, n_layers = 2, kernel_size=(3,3)):
        
        filters = n_filters
        ## Left branch of Y network
        left_inputs = Input(shape=input_shape)
        x = left_inputs ## Motion
        
        ## Right branch of Y network
        right_inputs = Input(shape=input_shape)
        y = right_inputs  ## Appearance

        # 2 blocks of ConvLSTM2D-AveragePooling3D/2D
        for i in range(n_layers):           
            ## Second block
            if i == n_layers-1 :
                ## Motion stream
                x = ConvLSTM2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               activation='relu',
                               data_format = 'channels_last',
                               return_sequences = True)(x)
                x = BatchNormalization()(x)
                x = ConvLSTM2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               activation='relu',
                               data_format = 'channels_last',
                               return_sequences = False)(x)
                x = BatchNormalization()(x)
                x = Dropout(0.25)(x)
                x = AveragePooling2D(pool_size=(2,2))(x)
                
                ## Appearance stream (Produces Attention Mask)
                y = ConvLSTM2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               activation='relu',
                               data_format = 'channels_last',
                               return_sequences = True)(y)
                y = BatchNormalization()(y)
                y = ConvLSTM2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               activation='relu',
                               data_format = 'channels_last',
                               return_sequences = False)(y)
                y = BatchNormalization()(y)
                y = Dropout(0.25)(y)
                y = AveragePooling2D(pool_size=(2,2))(y)
                
                ## Attention Mask 2
                
                mask = Conv2D(filters=1,
                        kernel_size=(1,1),
                        padding='same',
                        activation='sigmoid')(y)
                
                B,_, H, W, = y.shape
                norm = tf.norm(mask, ord=1, axis=1)
                norm = tf.norm(norm, ord=1, axis=1)
                norm = tf.norm(norm, ord=1, axis=1)
                norm = tf.keras.layers.Reshape((1, 1, 1))(norm)
                mask = tf.math.divide(mask * H * W, norm)
        


            ## First block   
            else:
                ## Motion
                x = ConvLSTM2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               activation='relu',
                               kernel_initializer='he_normal',
                               data_format = 'channels_last',
                               return_sequences = True)(x)
                x = BatchNormalization()(x)
                x = ConvLSTM2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               activation='relu',
                               data_format = 'channels_last',
                               return_sequences = True)(x)
                x = BatchNormalization()(x)
                x = Dropout(0.25)(x)
                x = AveragePooling3D(pool_size=(1,2,2))(x)
            
                ## Appearance (Attention Mask)
                y = ConvLSTM2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               activation='relu',
                               data_format = 'channels_last',
                               return_sequences = True)(y)
                y = BatchNormalization()(y)
                ## to feed forward
                

                ## to get attention mask
                y_ = ConvLSTM2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               activation='relu',
                               data_format = 'channels_last',
                               return_sequences = False)(y)
                y_ = BatchNormalization()(y_)
                y_ = Dropout(0.25)(y_)
                y_ = AveragePooling2D(pool_size=(2,2))(y_)

                y = ConvLSTM2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               activation='relu',
                               data_format = 'channels_last',
                               return_sequences = True)(y)
                y = BatchNormalization()(y)
                y = Dropout(0.25)(y)
                y = AveragePooling3D(pool_size=(1,2,2))(y)
                
                ## Attention Mask 1
                mask = Conv2D(filters=1,
                        kernel_size=(1,1),
                        padding='same',
                        activation='sigmoid')(y_)
            
                ## Attention Mask    
                B,T,_, H, W, = y.shape
                norm = tf.norm(mask, ord=1, axis=1)
                norm = tf.norm(norm, ord=1, axis=1)
                norm = tf.norm(norm, ord=1, axis=1)
                norm = tf.keras.layers.Reshape((T,1, 1, 1))(norm)
                mask = tf.math.divide(mask * T * H * W, norm)
                
            x = tf.math.multiply(x,mask, name ='Elementwise Multiplication')
        # Feature maps to vector before connecting to Dense 
        x = Flatten()(x)
        x = Dense(128,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        outputs = Dense(timesteps)(x)
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
        x = left_inputs ## Motion
        
        ## Right branch of Y network
        right_inputs = Input(shape=input_shape)
        y = right_inputs  ## Appearance

        ## 2 blocks 
        for i in range(n_layers):

            ## Motion 
            x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation='tanh')(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation='tanh')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.25)(x)
            x = AveragePooling2D(pool_size=(2,2))(x)
                
            ## Appearance
            y = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   padding='same',
                   activation='tanh')(y)
            y = BatchNormalization()(y)
            y = Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same',
                    activation='tanh')(y)
            y = BatchNormalization()(y)
            y = Dropout(0.25)(y)
            y = AveragePooling2D(pool_size=(2,2))(y)
                
            mask = Conv2D(filters=1,
                        kernel_size=(1,1),
                        padding='same',
                        activation='sigmoid')(y)
            
            ## Attention Mask    
            B,_, H, W, = y.shape
            norm = tf.norm(mask, ord=1, axis=1)
            norm = tf.norm(norm, ord=1, axis=1)
            norm = tf.norm(norm, ord=1, axis=1)
            norm = tf.keras.layers.Reshape((1, 1, 1))(norm)
            mask = tf.math.divide(mask * H * W, norm)
            x = tf.math.multiply(x,mask, name ='Elementwise Multiplication')
            filters *= 2

        # Feature maps to vector before connecting to Dense 
        x = Flatten()(x)
        x = Dense(128,activation='tanh')(x)
        outputs = Dense(1)(x)
        # Build the model (functional API)
        model = Model([left_inputs, right_inputs], outputs, name='DeepPhys') 
        return model