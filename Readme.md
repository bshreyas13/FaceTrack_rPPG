# FaceTrack_rPPG


Dataset Used: DEAP 

First copy dataset locally on drive. Then start by preprocessing the data. usage as shown below.

```shell
python preprocess_data.py -ds "path_to_Video Directory organized by subject" -lp " path_to_Label.mat"

```
## Update details

Modules Updated: tfrecordhandler
This module has all the tools and methods needed to write a TF Record file for an image sequence producing a TF Dataset Batch dataset object of (batch_size, timesteps, img_hieght, img_width, img_dept).
This can be used to train the ConvLSTM.

Script added: test_tfr.py

This scrip can be used as reference for usage of TFRHandler Class.

```shell
python test_tfr.py -id " input video1,input_video2" -bs "batch_size"

```
Model can be built, trained and tested using the build_train_model.py 

```shell
python build_train_model.py -m "FaceTrack_rPPG/DeepPhys" -ap " path to frame extracted roi directory" -mp "path to frame extracted ND directory" -lp "path tp labels by video directory"

```
Note: Data handling for Deep Phys is till being designed as the python generator causes memeory leaks and results in the process being killed 