# FaceTrack_rPPG

## Languages

Python3

##Dataset Used: 

DEAP 

First copy dataset locally on drive. Then start by preprocessing the data. usage as shown below.

```shell
python preprocess_data.py -ds "path_to_Video Directory organized by subject" -lp " path_to_Label.mat"

```
## Update details

Stable update for DeepPhys. Can be used to tune hyperparameters and improve network performance.

FaceTrack_rPPG is a large model that still breaks when n_filters is large, testing to find optimum settings for stability

## Modules and Scripts

This package contains the following modules:
	* preproccessor : Class of tools used in prepprocess_data.py.
	* tfrecordhandler_FTR/tfrecordhandler_DP : Class of tools to read and write TFRecord.
	* models : contains tensorflow model built using tf functional API
	* videodatasethandler : Class contains a tool to check dataset integrity. Also contains an unstable version of datagenerator based input pipeline.

Scripts:
	* test_tfr.py : To test the tfrecordhandler and display some examples. Can be used as reference for usage of  Class. Usage as shown below.


```shell
python test_tfr.py -m "FaceTrack_rPPG/DeepPhys" -id " input video1,input_video2" -bs "batch_size"

```

	* prepare_data_v2 : Reads TFrecord to produce tf.Data dataset. Also contains auxilary tools to suffling data on video level and fixing label filenames.
	* build_train_model_v2.py  : To build compile the tf models, parse dtatsets and train network. Usage as shown below.

```shell
python build_train_model.py -m "FaceTrack_rPPG/DeepPhys" -ap " path to frame extracted roi directory" -mp "path to frame extracted ND directory" -lp "path tp labels by video directory" -wtxt -wtfr -rmtxt -rmtfr

```
### args
	** "-m","--model", required = True , help = "FaceTrack_rPPG or DeepPhys"
    ** "-ap","--appearance", required = True , help = "Path to Apperance Stream Data"
    ** "-mp","--motion", required = True , help = "Path to Motion Stream Data")
    ** "-lp","--labels", required = True , help = "Path to  Label by video"
    ** "-wtxt","--write_textfiles", action ='store_true',required = False , help = "Flag to enable/disable data txt file writing "
    ** "-wtfr","--write_tfrecords", action ='store_true',required = False , help = "Flag to enable/disable data TF Records "
    ** "-ts","--timesteps", required = False , help = "timestep for FaceTrack_rPPG, defaults to 5"
    ** "-flf","--fix_label_filenames", action ='store_true',required = False , help = "Flag to enable fix for label filenames in case they are missing preceeding zeros in sXX_trialXX"
    ** "-cdi","--check_data_integrity", action ='store_true',required = False , help = "Flag to check the count of images in each folder (sXX_trialXX)"
    ** "-rmtxt","--remove_textfiles", action ='store_true',required = False , help = "Flag to remove txt files from previous runs")
    ** "-rmtfr","--remove_tfrecords", action ='store_true',required = False , help = "Flag to remove tfrecords from previous runs"

