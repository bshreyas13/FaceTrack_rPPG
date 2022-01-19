# FaceTrack_rPPG

## Languages and Dependencies

Python3
Tensorflow 2 or above

All dependencies can be install using the requirements.txt file with pip.

## Dataset Used: 

DEAP : A Database for Emotion Analysis using Physiological Signals (PDF)

"DEAP: A Database for Emotion Analysis using Physiological Signals (PDF)", S. Koelstra, C. Muehl, M. Soleymani, J.-S. Lee, A. Yazdani, T. Ebrahimi, T. Pun, A. Nijholt, I. Patras, EEE Transactions on Affective Computing, vol. 3, no. 1, pp. 18-31, 2012

First copy dataset locally on drive. Then start by preprocessing the data. usage as shown below.

```shell
python preprocess_data.py -ds "path_to_Video Directory organized by subject" -lp " path_to_Label.mat"

```
## Update details

Stable update for both models. Can be used to tune hyperparameters and improve network performance.
Flags added to save model and retrain model in case running multiple epochs at once isn't possible. Model is saved after every epoch '../saved_models' along with a metric.npy under '../Metric_Files' for very training run. 

Tools to load metrics and plot overall training curve are being added to helper_funcs. 

**Note:**FaceTrack_rPPG is a still breaks when n_filters is large, default uses 32 filters in both blocks.

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
    ** "-mp","--motion", required = True , help = "Path to Motion Stream Data"
    ** "-lp","--labels", required = True , help = "Path to  Label by video"
    ** "-tfr_path","--tfrecord_path", required = False , help = "Alternate TFRecords path if needed"
    ** "-wtxt","--write_textfiles", action ='store_true',required = False , help = "Flag to enable/disable data txt file writing "
    ** "-wtfr","--write_tfrecords", action ='store_true',required = False , help = "Flag to enable/disable data TF Records "
    ** "-ts","--timesteps", required = False , help = "timestep for FaceTrack_rPPG, defaults to 5"
    ** "-flf","--fix_label_filenames", action ='store_true',required = False , help = "Flag to enable fix for label filenames in case they are missing preceeding zeros in sXX_trialXX"
    ** "-cdi","--check_data_integrity", action ='store_true',required = False , help = "Flag to check the count of images in each folder (sXX_trialXX)"
    ** "-rmtxt","--remove_textfiles", action ='store_true',required = False , help = "Flag to remove txt files from previous runs"
    ** "-rmtfr","--remove_tfrecords", action ='store_true',required = False , help = "Flag to remove tfrecords from previous runs"
    ** "-tpu","--run_on_tpu", action ='store_true',required = False , help = "Flag to enable run on TPU"
    ** "-n_train","--no_training", action ='store_true',required = False , help = "Flag to enable run only write TFRecords without training"
    ** "-lm_train","--load_model_train", action ='store_true',required = False , help = "Flag to enableloading a model and continue training"
    ** "-lm_path","--load_model_path",required = False , help = "Path to model"

## Usage Guide

Use this [colab notebook](https://colab.research.google.com/drive/18r3XX-IJUR-aF4c_93pB2RNOIVLP4GZK#scrollTo=ceca9bNXX-4A) as reference to setup and run the package.
Refer Notes in code for more details.
In case of queries email me and I will get in touch with you as soon as possible.


