# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:19:24 2021

@author: bshreyas
"""
import ArgumentParser as ap
import os
import numpy as np
import tensorflow as tf
from tf.keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from modules.models import Models
from tensorflow.keras.utils import plot_model


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
def train_test_plot(model, train_ds,val_ds,test_ds,epochs,batch_size):
  
  
  # Compile model

  model.compile(loss='mse',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
  
  # prepare model model saving directory.
  save_dir = os.path.join(os.path.dirname(os.getcwd(), 'saved_models'))
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
  history= model.fit(train_ds, batch_size=batch_size,
                        validation_data=val_ds,
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)
 

  # Evaluate Model on Test set
  score = model.evaluate(test_ds,
                       batch_size=batch_size,
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
    
    