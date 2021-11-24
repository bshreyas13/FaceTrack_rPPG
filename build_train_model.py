# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:19:24 2021

@author: bshreyas
"""
import argparse as ap
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from modules.models import Models
from tensorflow.keras.utils import plot_model
import prepare_data as prep
from modules.videodatasethandler import VideoDatasetHandler

##Learning Rate Schedule ##
def lr_schedule(epoch):

    lr = 1e-3
    if epoch > 80:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
        
    print('Learning rate: ', lr)
    return lr

## Funtion to train , test and plot training curve ##
def train_test_plot(model,optimizer, train_ds,val_ds,test_ds,epochs,batch_size):
  
  
  # Compile model

  model.compile(loss='mse',
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'], run_eagerly=False)
  
  # prepare model model saving directory.
  save_dir = os.path.join(os.path.dirname(os.getcwd()), 'saved_models')
  model_name = 'FaceTrack_rPPG_model.{epoch:03d}.h5' 
  if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
  filepath = os.path.join(save_dir, model_name)

  # prepare callbacks for model saving and for learning rate adjustment.
  
  checkpoint = ModelCheckpoint(filepath=filepath,
                               monitor='val_acc',
                               verbose=5,
                               save_best_only=True)

  lr_scheduler = LearningRateScheduler(lr_schedule)

  lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

  callbacks = [checkpoint, lr_reducer, lr_scheduler]
  
  # Train the model 
  history= model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)
 
  # Evaluate Model on Test set
  score = model.evaluate(test_ds,
                       verbose=2)
  print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))
  
  #Plot training curve
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()
  
if __name__ == '__main__':
    
    parser = ap.ArgumentParser()
    parser.add_argument("-m","--model", required = True , help = "Facetrack_rPPG or DeepPhys")
    parser.add_argument("-ap","--appearance", required = True , help = "Path to Apperance Stream Data")
    parser.add_argument("-mp","--motion", required = True , help = "Path to Motion Stream Data")
    parser.add_argument("-lp","--labels", required = True , help = "Path to Motion Stream Data")
    parser.add_argument("-ts","--timesteps", required = False , help = "timestep for FaceTrack_rPPG, defaults to 5")
    
    args = vars(parser.parse_args())
    
    model = args["model"]
    if args["timesteps"] == None:    
        timesteps = 5
    else:
        timesteps = args["timesteps"]
    appearance_path = args["appearance"]
    motion_path = args["motion"]
    labels_path = args["labels"]
    
    
    n_filters =32
    batch_size = 10
    epochs = 2
    subset=0.01
    val_split=0.1
    test_split=0.2
    vdh = VideoDatasetHandler()
    if model == 'FaceTrack_rPPG':
            input_shape = (timesteps,300,215,3)
            x_shape = input_shape
            y_shape = (timesteps,)
            model= Models.FaceTrack_rPPG(input_shape, timesteps, n_filters)
           # verify the model using graph
            plot_model(model, to_file='FaceTrack_rPPG.png', show_shapes=True)
            model.summary()
    elif model == 'DeepPhys':
            input_shape = (300,215,3)
            x_shape = input_shape
            y_shape = ()
            model= Models.DeepPhys(input_shape, n_filters)
           # verify the model using graph
            plot_model(model, to_file='DeepPhys.png', show_shapes=True)
            model.summary()

    ## Get data, prepare and optimize it for Training and tetsing ##
    train_ds,val_ds,test_ds = prep.getDatasets(model, appearance_path,motion_path, labels_path, x_shape, y_shape,batch_size , timesteps,subset,val_split,test_split)
    ## Normalize Data
    # train_ds = prep.addNormalizationLayer(train_ds)
    # val_ds = prep.addNormalizationLayer(val_ds)
    # test_ds = prep.addNormalizationLayer(test_ds)
    
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)
    
    ## TF Performance Configuration
    try:
      AUTOTUNE = tf.data.AUTOTUNE     
    except:
      AUTOTUNE = tf.data.experimental.AUTOTUNE 
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    print(train_ds.element_spec)
    ## Call train_test_plot to start the process
    optimizer = Adam
    train_test_plot(model,optimizer, train_ds,val_ds,test_ds,epochs,batch_size)
    
    
    