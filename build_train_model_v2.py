# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:19:24 2021

@author: bshreyas
"""
## Update Notes ##
## DeepPhys stable , tuning network for performance  ##
## FaceTrack_rPPG is stable with bactch_size < 5 on colab ##
## Since colab runtimes are limited, added flags to load a saved model and continue trianing ##
## However lr scheduler wont work in that case and lr has to manually updated with each iteration ##
## plot_model buggy after attention mask update ##
#######################################
## Script to build and train models  ##
#######################################

import argparse as ap
import os
import sys
import numpy as np
import pathlib
import tensorflow as tf
import shutil
import json
from tensorflow.keras.optimizers import Adam, Adadelta,RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from modules.models import Models
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import prepare_data_v2 as prep
from modules.videodatasethandler import VideoDatasetHandler
from modules.preprocessor import Preprocessor

##Learning Rate Schedule ##
def lr_schedule(epoch):

    lr = 0.5e-2
    if epoch > 80:
       lr *= 1e-1
    elif epoch > 60:
        lr *= 0.5
    elif epoch > 40:
        lr *= 1e-1
    elif epoch > 10:
        lr *= 0.5
        
    print('Learning rate: ', lr)
    return lr



## Function to train , test and plot training curve ##
def train_test_plot(model,model_name_, train_ds,val_ds,test_ds,epochs,batch_size):
  
    # prepare model model saving directory.
    save_dir = os.path.join(os.path.dirname(os.getcwd()), 'saved_models', model_name_)
    model_name = 'saved_{epoch:03d}.h5' 
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # prepare callbacks for model saving and for learning rate adjustment.
  
    checkpoint = ModelCheckpoint(filepath=filepath,
                               monitor='val_loss',
                               verbose=5)

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
  
    
    save_metric_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Metric_Files',model_name_))
    save_metric_path.mkdir(parents=True,exist_ok=True)
    suffix = 0
    for file in os.listdir(save_metric_path):
        if file.startswith('metrics'):
            suffix += 1
    suffix = str(suffix)
    save_metric_file = os.path.join(save_metric_path.as_posix(),'metrics'+suffix+'.npy')
    np.save(save_metric_file,history.history)

  
    # Evaluate Model on Test set
    score = model.evaluate(test_ds,
                       verbose=2)
    print("\nTest mse: {}".format(score[1]))
  
    
  
if __name__ == '__main__':
    
    parser = ap.ArgumentParser()
    parser.add_argument("-m","--model", required = True , help = "FaceTrack_rPPG or DeepPhys")
    parser.add_argument("-ap","--appearance", required = True , help = "Path to Apperance Stream Data")
    parser.add_argument("-mp","--motion", required = True , help = "Path to Motion Stream Data")
    parser.add_argument("-lp","--labels", required = True , help = "Path to  Label by video")
    parser.add_argument("-tfr_path","--tfrecord_path", required = False , help = "Alternate TFRecords path if needed")
    parser.add_argument("-wtxt","--write_textfiles", action ='store_true',required = False , help = "Flag to enable/disable data txt file writing ")
    parser.add_argument("-wtfr","--write_tfrecords", action ='store_true',required = False , help = "Flag to enable/disable data TF Records ")
    parser.add_argument("-ims", "--image_size", required = False , help = "Desired input img size.")
    parser.add_argument("-ts","--timesteps", required = False , help = "timestep for FaceTrack_rPPG, defaults to 5")
    parser.add_argument("-bs", "--batch_size", required = False , help = "Desired batch size. Defaults to 2 for FTR and 10 for DeepPhys")
    parser.add_argument("-ep", "--epochs", required = False , help = "Desired number of epochs for training. Defaults to 2 ")
    parser.add_argument("-n_blks", "--num_blocks", required = False , help = "Desired number of blocks of ConvLSTM2D/Conv2D and Average pooling layers . Defaults to 1 for FTR and 2 for DeepPhys ")
    parser.add_argument("-n_fltrs", "--num_filters", required = False , help = "Desired number of filters for ConvLSTM2D/Conv2D layers . Defaults to 16 for FTR and 32 for DeepPhys ")
    parser.add_argument("-sbst", "--subset", required = False , help = "Desired subset of Deap dataset. Defaults to 0.2 for FTR and 0.5 for DeepPhys ")
    parser.add_argument("-sbst_train", "--subset_to_train", required = False , help = "Desired subset of Deap dataset. Defaults to 0.1 for DeepPhys ")
    parser.add_argument("-fxlnm","--fix_label_filenames", action ='store_true',required = False , help = "Flag to enable fix for label filenames in case they are missing preceeding zeros in sXX_trialXX")
    parser.add_argument("-chkdt","--check_data_integrity", action ='store_true',required = False , help = "Flag to check the count of images in each folder (sXX_trialXX)")
    parser.add_argument("-rmtxt","--remove_textfiles", action ='store_true',required = False , help = "Flag to remove txt files from previous runs")
    parser.add_argument("-rmtfr","--remove_tfrecords", action ='store_true',required = False , help = "Flag to remove tfrecords from previous runs")
    parser.add_argument("-tpu","--run_on_tpu", action ='store_true',required = False , help = "Flag to enable run on TPU")
    parser.add_argument("-n_train","--no_training", action ='store_true',required = False , help = "Flag to enable run only write TFRecords without training")
    parser.add_argument("-lm_train","--load_model_train", action ='store_true',required = False , help = "Flag to enableloading a model and continue training")
    parser.add_argument("-lm_path","--load_model_path",required = False , help = "Path to model")
    parser.add_argument("-sa","--spatial_average", action ='store_true', required = False , help = "Toggle to enable spatial_averaging")
    
    args = vars(parser.parse_args())
    
    
    ## Get args
    model = args["model"]
    wtxt = args["write_textfiles"]
    wtfr = args["write_tfrecords"]
    rmtxt = args["remove_textfiles"]
    rmtfr = args["remove_tfrecords"]   
    tpu = args["run_on_tpu"]
    n_train = args["no_training"]
    lm_train = args['load_model_train']
    spatial_avg = args["spatial_average"]
    img_size = args["image_size"]
    if tpu == True:
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
            print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
        except ValueError:
            raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        tpu_strategy = tf.distribute.TPUStrategy(tpu)

    if args["timesteps"] == None:    
        timesteps = 5
    else:
        timesteps = int(args["timesteps"])
    
    appearance_path = args["appearance"]
    motion_path = args["motion"]
    labels_path = args["labels"]
    tfr_path = args["tfrecord_path"]
    
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
    
    
   
    ## Model conditioning , to pick correct data and model ##
    ## Face_Track_rPPG ##
    if model == "FaceTrack_rPPG":
        
        model_name = "FaceTrack_rPPG"
        print("Building and Training {}".format(model_name))
        
        ##get args
        if args["num_blocks"] == None:
            n_layers = 1
        else:
            n_layers = int(args["num_blocks"])
        
        if args["num_filters"] == None:
            n_filters = 16 
        else:
            n_filters = int(args["num_filters"])
        if args["batch_size"] == None:    
            batch_size = 2
        else:
            batch_size = int(args["batch_size"])
        
        if args["epochs"] == None:    
            epochs = 2 
        else:
            epochs = int(args["epochs"])
        if args["subset"] == None:    
            subset=0.2
            print("Using {}% of the dataset.".format(subset*100))
        else:
            subset = float(args["subset"])
            print("Using {}% of the dataset.".format(subset*100))
            ## Ensure subset is large enough to produce at least 1 val , test videos ##
            ## Handling for this corner case is not yet added ##
        if args["subset_to_train"] == None:    
            subset_train=1
        else:
            subset_train = float(args["subset_to_train"])## Ensure subset is large enough to produce at least 1 val , test videos ##
            ## Handling for this corner case is not yet added ##
        if img_size == None:
            img_size = "215X300X3"
        else:
            img_size = args["image_size"]
        
        img_size = [int(dim) for dim in img_size.split('X')]
        img_size = (img_size[0],img_size[1],img_size[2])
        val_split=0.1
        test_split=0.2
        
        ## Remove folder from previous run if any , controlled bu flags    
        if rmtxt == True :
            shutil.rmtree(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Txt', model_name))
    
        if spatial_avg == True:
            p = Preprocessor()
            ## Cycle through rgb and ND images nad resixe nd save as jpeg
            videos = os.listdir(appearance_path)
            for video in videos:
                
                p.resizeAndGetND(appearance_path, video,motion_path, img_size = img_size)
        
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
        
        if tfr_path == None:    
            tfrecord_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'TFRecords',model_name))
        else:
            tfrecord_path = pathlib.Path(tfr_path)

        tfrecord_path.mkdir(parents=True,exist_ok=True)
        
            
        input_shape = (timesteps,img_size[0],img_size[1],img_size[2])
        optimizer = Adam(learning_rate=lr_schedule(0))
        if tpu == True:
            with tpu_strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
  
                model= Models.FaceTrack_rPPG(input_shape, timesteps, n_filters,n_layers=n_layers)    
                # Compile model
                model.compile(loss='mse',
                            optimizer= optimizer,
                            metrics=['mae'], run_eagerly=False)
        
        else:
            model= Models.FaceTrack_rPPG(input_shape, timesteps, n_filters,n_layers=n_layers)    
            # Compile model
            model.compile(loss='mse',
                        optimizer= optimizer,
                        metrics=['mae'], run_eagerly=False)
        #verify the model using graph
        #plot_model(model, to_file='FaceTrack_rPPG.png', show_shapes=True) ## plot model currently buggy 
        model.summary()

        ## Get data, prepare and optimize it for Training and tetsing ##
        train_ds,val_ds,test_ds = prep.getDatasets(model_name,appearance_path,motion_path,labels_path,txt_files_paths,tfrecord_path, img_size, batch_size=batch_size, timesteps=timesteps, subset=subset, subset_read= subset_train,val_split = val_split , test_split =test_split,write_txt_files=wtxt, create_tfrecord=wtfr, rot = 1)
        

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
        
        ## If no training only TF record generation enable -n_train ##
        if n_train == True:
            sys.exit()

        ## To load and continue training ##
        if lm_train == True:
            try:
                lm_path = args["load_model_path"]
                model = load_model(lm_path)
                print("Loaded model at: {}".format(lm_path))
                print("Continuing training for loaded model")
            except:
                print("Specify load model path with -lm_path if -lm_train flag is active ")        
        ## Call train_test_plot to start the process
        train_test_plot(model, model_name, train_ds,val_ds,test_ds,epochs,batch_size)
   
    ## DeepPhys ##
    elif model == "DeepPhys":
        
        model_name = "DeepPhys"
        print("Building and Training {}".format(model_name))
        
        ##get args
        if args["num_blocks"] == None:
            n_layers = 1
        else:
            n_layers = int(args["num_blocks"])
        
        if args["num_filters"] == None:
            n_filters = 32 
        else:
            n_filters = int(args["num_filters"])
        if args["batch_size"] == None:    
            batch_size = 10
        else:
            batch_size = int(args["batch_size"])
        
        if args["epochs"] == None:    
            epochs = 2 
        else:
            epochs = int(args["epochs"])
        if args["subset"] == None:    
            subset=0.2 
        else:
            subset = float(args["subset"])## Ensure subset is large enough to produce at least 1 val , test videos ##
            ## Handling for this corner case is not yet added ##
        if args["subset_to_train"] == None:    
            subset_train=0.1 
        else:
            subset_train = float(args["subset_to_train"])## Ensure subset is large enough to produce at least 1 val , test videos ##
            ## Handling for this corner case is not yet added ##
        if img_size == None:
            img_size = "215X300X3"
        else:
            img_size = args["image_size"]
        
        img_size = [int(dim) for dim in img_size.split('X')]
        img_size = (img_size[0],img_size[1],img_size[2])
        val_split=0.1
        test_split=0.2
        
        
        ## Remove folder from previous run if any , controlled bu flags    
        if rmtxt == True :
            shutil.rmtree(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Txt', model_name))
    
        if spatial_avg == True:
            p = Preprocessor()
            ## Cycle through rgb and ND images nad resixe nd save as jpeg
            videos = os.listdir(appearance_path)
            for video in videos:             
                p.resizeAndGetND(appearance_path, video,motion_path, img_size = img_size)
                
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
           
        if tfr_path == None:    
            tfrecord_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'TFRecords',model_name))
        else:
            tfrecord_path = pathlib.Path(tfr_path)

        tfrecord_path.mkdir(parents=True,exist_ok=True)
            
        input_shape = (img_size[0],img_size[1],img_size[2]) 
        optimizer = Adadelta(learning_rate=lr_schedule(0))
        if tpu == True:
            with tpu_strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
  
                model= Models.DeepPhys(input_shape, n_filters)    
                # Compile model
                model.compile(loss='mse',
                            optimizer= optimizer,
                            metrics=['mae'], run_eagerly=False)
        
        else:
            model = Models.DeepPhys(input_shape, n_filters)
            # Compile model
            model.compile(loss='mse',
                        optimizer= optimizer,
                        metrics=['mae'], run_eagerly=False)
        
        #verify the model using graph
        #plot_model(model, to_file='DeepPhys.png', show_shapes=True) ## Plot model is currenlty failing 
        model.summary()

        ## Get data, prepare and optimize it for Training and tetsing ##
        train_ds,val_ds,test_ds = prep.getDatasets(model_name,appearance_path,motion_path,labels_path,txt_files_paths,tfrecord_path, img_size, batch_size=batch_size, timesteps=timesteps, subset=subset, subset_read = subset_train, val_split = val_split , test_split =test_split,write_txt_files=wtxt, create_tfrecord=wtfr,rot=1)
   
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

        if n_train == True:
            sys.exit()

        if lm_train == True:
            try:
                lm_path = args["load_model_path"]
                model = load_model(lm_path)
                print("Loaded model at: {}".format(lm_path))
                print("Continuing training for loaded model")
            except:
                print("Specify load model path with -lm_path if -lm_train flag is active ")
        ## Call train_test_plot to start the process
        train_test_plot(model, model_name, train_ds,val_ds,test_ds,epochs,batch_size)
    
    ## Vanilla DeepPhys ##
    elif model == "DeepPhys_V0":
        
        model_name = "DeepPhys_V0"
        print("Building and Training {}".format(model_name))
        
        ##get args
        if args["num_blocks"] == None:
            n_layers = 1
        else:
            n_layers = int(args["num_blocks"])
        
        if args["num_filters"] == None:
            n_filters = 32 
        else:
            n_filters = int(args["num_filters"])
        if args["batch_size"] == None:    
            batch_size = 10
        else:
            batch_size = int(args["batch_size"])
        
        if args["epochs"] == None:    
            epochs = 2 
        else:
            epochs = int(args["epochs"])
        if args["subset"] == None:    
            subset=0.2 
        else:
            subset = float(args["subset"])## Ensure subset is large enough to produce at least 1 val , test videos ##
            ## Handling for this corner case is not yet added ##
        if args["subset_to_train"] == None:    
            subset_train=0.1 
        else:
            subset_train = float(args["subset_to_train"])## Ensure subset is large enough to produce at least 1 val , test videos ##
            ## Handling for this corner case is not yet added ##
        if img_size == None:
            img_size = "36X36X36"
        else:
            img_size = args["image_size"]
        
        img_size = [int(dim) for dim in img_size.split('X')]
        img_size = (img_size[0],img_size[1],img_size[2])
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
           
        if tfr_path == None:    
            tfrecord_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'TFRecords',model_name))
        else:
            tfrecord_path = pathlib.Path(tfr_path)

        tfrecord_path.mkdir(parents=True,exist_ok=True)
            
        input_shape = (img_size[0],img_size[1],img_size[2]) 
        optimizer = Adadelta(learning_rate=lr_schedule(0))
        if tpu == True:
            with tpu_strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
  
                model= Models.DeepPhys(input_shape, n_filters)    
                # Compile model
                model.compile(loss='mse',
                            optimizer= optimizer,
                            metrics=['mae'], run_eagerly=False)
        
        else:
            model = Models.DeepPhys(input_shape, n_filters)
            # Compile model
            model.compile(loss='mse',
                        optimizer= optimizer,
                        metrics=['mae'], run_eagerly=False)
        
        #verify the model using graph
        #plot_model(model, to_file='DeepPhys.png', show_shapes=True) ## Plot model is currenlty failing 
        model.summary()

        ## Get data, prepare and optimize it for Training and tetsing ##
        train_ds,val_ds,test_ds = prep.getDatasets(model_name,appearance_path,motion_path,labels_path,txt_files_paths,tfrecord_path, img_size, batch_size=batch_size, timesteps=timesteps, subset=subset, subset_read = subset_train, val_split = val_split , test_split =test_split,write_txt_files=wtxt, create_tfrecord=wtfr,rot=1)
   
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

        if n_train == True:
            sys.exit()

        if lm_train == True:
            try:
                lm_path = args["load_model_path"]
                model = load_model(lm_path)
                print("Loaded model at: {}".format(lm_path))
                print("Continuing training for loaded model")
            except:
                print("Specify load model path with -lm_path if -lm_train flag is active ")
        ## Call train_test_plot to start the process
        train_test_plot(model, model_name, train_ds,val_ds,test_ds,epochs,batch_size)
    