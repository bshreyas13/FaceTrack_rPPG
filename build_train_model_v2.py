# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:19:24 2021

@author: bshreyas
"""
## Update Notes ##
## DeepPhys stable , tuning network for performance  ##
## FaceTrack rPPG , only works with n_layers = 1 , OOM kills model ## 
## n_layers > 1 crashes whiel compiling the model ##
## n_filters = 32 kills training with OOM ##
## Mirrored Strategy to be tested for improved training speed ##

#######################################
## Script to build and train models  ##
#######################################

import argparse as ap
import os
import numpy as np
import pathlib
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import Adam, Adadelta,RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from modules.models import Models
from tensorflow.keras.utils import plot_model
import prepare_data_v2 as prep
from modules.videodatasethandler import VideoDatasetHandler


##Learning Rate Schedule ##
def lr_schedule(epoch):

    lr = 1e-1
    if epoch > 80:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 0.5e-2
    elif epoch > 20:
        lr *= 1e-2
        
    print('Learning rate: ', lr)
    return lr

## Function to train , test and plot training curve ##
def train_test_plot(model,model_name,optimizer, train_ds,val_ds,test_ds,epochs,batch_size):
  
  
  # Compile model

  model.compile(loss='mse',
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'], run_eagerly=False)
  
  # prepare model model saving directory.
  save_dir = os.path.join(os.path.dirname(os.getcwd()), 'saved_models', model_name)
  model_name = 'saved_{epoch:03d}.h5' 
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
                        epochs=epochs, verbose=1, workers=4,batch_size=batch_size,
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
  plt.savefig('Train_cruve_{}.jpg'.format(model_name))
  
if __name__ == '__main__':
    
    parser = ap.ArgumentParser()
    parser.add_argument("-m","--model", required = True , help = "FaceTrack_rPPG or DeepPhys")
    parser.add_argument("-ap","--appearance", required = True , help = "Path to Apperance Stream Data")
    parser.add_argument("-mp","--motion", required = True , help = "Path to Motion Stream Data")
    parser.add_argument("-lp","--labels", required = True , help = "Path to  Label by video")
    parser.add_argument("-wtxt","--write_textfiles", action ='store_true',required = False , help = "Flag to enable/disable data txt file writing ")
    parser.add_argument("-wtfr","--write_tfrecords", action ='store_true',required = False , help = "Flag to enable/disable data TF Records ")
    parser.add_argument("-ts","--timesteps", required = False , help = "timestep for FaceTrack_rPPG, defaults to 5")
    parser.add_argument("-flf","--fix_label_filenames", action ='store_true',required = False , help = "Flag to enable fix for label filenames in case they are missing preceeding zeros in sXX_trialXX")
    parser.add_argument("-cdi","--check_data_integrity", action ='store_true',required = False , help = "Flag to check the count of images in each folder (sXX_trialXX)")
    parser.add_argument("-rmtxt","--remove_textfiles", action ='store_true',required = False , help = "Flag to remove txt files from previous runs")
    parser.add_argument("-rmtfr","--remove_tfrecords", action ='store_true',required = False , help = "Flag to remove tfrecords from previous runs")
    
    args = vars(parser.parse_args())
    
    
    ## Get args
    model = args["model"]
    wtxt = args["write_textfiles"]
    wtfr = args["write_tfrecords"]
    rmtxt = args["remove_textfiles"]
    rmtfr = args["remove_tfrecords"]
  
    if args["timesteps"] == None:    
        timesteps = 5
    else:
        timesteps = int(args["timesteps"])
    appearance_path = args["appearance"]
    motion_path = args["motion"]
    labels_path = args["labels"]
    
    
    if args["fix_label_filenames"] == True:
        prep.fixLabelFilenames(labels_path)
    
    if args["check_data_integrity"] == True:
        vdh = VideoDatasetHandler()
        incomp_appearance,_ = vdh.verifyDataset(appearance_path)
        incomp_motion,_ = vdh.verifyDataset(motion_path)
        print("No of incompletely extracted appearance stream examples:",len(incomp_appearance))
        print("No of incompletely extracted motion stream examples:",len(incomp_motion))
        print("Droping incomplete examples")
        for folder in incomp_appearance:
            if os.path.isdir(os.path.join(appearance_path,folder)):
                shutil.rmtree(os.path.join(appearance_path,folder))
            elif os.path.isdir(os.path.join(motion_path,folder)):
                shutil.rmtree(os.path.join(motion_path,folder))
        for folder in incomp_motion:
            if folder not in incomp_appearance:
                if os.path.isdir(os.path.join(appearance_path,folder)):
                    shutil.rmtree(os.path.join(appearance_path,folder))
                elif os.path.isdir(os.path.join(motion_path,folder)):
                    shutil.rmtree(os.path.join(motion_path,folder))
        ap_list = os.listdir(appearance_path)
        mo_list = os.listdir(motion_path)
        print("No of data folders :{} Appearance ,{} Motion".format(len(ap_list),len(mo_list)))
    
    
   
    
    if model == "FaceTrack_rPPG":
        
        model_name = "FaceTrack_rPPG"
        print("Building and Training {}".format(model_name))
        
        n_layers = 1 
        n_filters = 16 
        batch_size = 2
        epochs = 20
        subset=0.2 ## Ensure subset is large enough to produce at least 1 val , test videos ##
        ## Handling for this corner case is not yet added ##
        val_split=0.1
        test_split=0.2
        
        ## Remove folder from previous run if any , controlled bu flags    
        if rmtxt == True :
            shutil.rmtree(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Txt', model_name))
    
        
        ## Check for txt file and tfrecord paths
        train_txt_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Txt', model_name, 'Train'))
        train_txt_path.mkdir(parents=True,exist_ok=True)
    
        val_txt_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Txt', model_name, 'Val'))
        val_txt_path.mkdir(parents=True,exist_ok=True)
    
        test_txt_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Txt', model_name, 'Test'))
        test_txt_path.mkdir(parents=True,exist_ok=True)
    
    
        ## create list of txt_file paths for getDataset ##
        txt_files_paths = [train_txt_path,val_txt_path,test_txt_path]
    
        if rmtfr == True :
            shutil.rmtree(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'TFRecords', model_name))
            
        tfrecord_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'TFRecords',model_name))
        tfrecord_path.mkdir(parents=True,exist_ok=True)
        
            
        input_shape = (timesteps,215,300,3)  
        model= Models.FaceTrack_rPPG(input_shape, timesteps, n_filters,n_layers=n_layers)    
        #verify the model using graph
        plot_model(model, to_file='FaceTrack_rPPG.png', show_shapes=True)
        model.summary()

        ## Get data, prepare and optimize it for Training and tetsing ##
        train_ds,val_ds,test_ds = prep.getDatasets(model_name,appearance_path,motion_path,labels_path,txt_files_paths,tfrecord_path, batch_size=batch_size, timesteps=timesteps, subset=subset, val_split = val_split , test_split =test_split,write_txt_files=wtxt, create_tfrecord=wtfr)
   
        ## Buffer size automation
        try:
            AUTOTUNE = tf.data.AUTOTUNE     
        except:
            AUTOTUNE = tf.data.experimental.AUTOTUNE 
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
        
        for (x_l,x_r),(y), in train_ds.take(1):    
            print('Appearance Input Shape:',x_r.shape)      
            print('Motion Input Shape',x_l.shape)
            print('Output',y.shape)
        ## Call train_test_plot to start the process
        optimizer = Adam
        train_test_plot(model, model_name, optimizer, train_ds,val_ds,test_ds,epochs,batch_size)
   
    elif model == "DeepPhys":
        
        model_name = "DeepPhys"
        print("Building and Training {}".format(model_name))
        
        n_filters = 32
        batch_size = 5
        epochs = 25
        subset=0.5 ## Ensure subset is large enough to produce at least 1 val , test videos ##
        ## Handling for this corner case is not yet added ##
        val_split=0.1
        test_split=0.2
        
        ## Remove folder from previous run if any , controlled bu flags    
        if rmtxt == True :
            shutil.rmtree(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Txt', model_name))
    
        
        ## Check for txt file and tfrecord paths
        train_txt_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Txt', model_name, 'Train'))
        train_txt_path.mkdir(parents=True,exist_ok=True)
    
        val_txt_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Txt', model_name, 'Val'))
        val_txt_path.mkdir(parents=True,exist_ok=True)
    
        test_txt_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Txt', model_name, 'Test'))
        test_txt_path.mkdir(parents=True,exist_ok=True)
    
    
        ## create list of txt_file paths for getDataset ##
        txt_files_paths = [train_txt_path,val_txt_path,test_txt_path]
    
        if rmtfr == True :
            shutil.rmtree(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'TFRecords',model_name))
            
        tfrecord_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'TFRecords',model_name))
        tfrecord_path.mkdir(parents=True,exist_ok=True)
            
        input_shape = (215,300,3)   
        model= Models.DeepPhys(input_shape, n_filters)
        #verify the model using graph
        plot_model(model, to_file='DeepPhys.png', show_shapes=True)
        model.summary()

        ## Get data, prepare and optimize it for Training and tetsing ##
        train_ds,val_ds,test_ds = prep.getDatasets(model_name,appearance_path,motion_path,labels_path,txt_files_paths,tfrecord_path, batch_size=batch_size, timesteps=timesteps, subset=subset, val_split = val_split , test_split =test_split,write_txt_files=wtxt, create_tfrecord=wtfr)
   
        ## TF Performance Configuration
        try:
            AUTOTUNE = tf.data.AUTOTUNE     
        except:
            AUTOTUNE = tf.data.experimental.AUTOTUNE 
        
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    
        for (x_l,x_r),(y), in train_ds.take(1):    
            print('Appearance Input Shape:',x_r.shape)      
            print('Motion Input Shape',x_l.shape)
            print('Output',y.shape)
        ## Call train_test_plot to start the process
        optimizer = Adadelta
        train_test_plot(model, model_name,optimizer, train_ds,val_ds,test_ds,epochs,batch_size)
    
    