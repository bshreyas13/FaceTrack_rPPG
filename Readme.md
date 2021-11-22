# FaceTrack_rPPG


Dataset Used: DEAP 

First copy dataset locally on drive. Then start by preprocessing the data. usage as shown below.

```shell
python preprocess_data.py -ds "path_to_Video Directory organized by subject" -lp " path_to_Label.mat"

```
## Update details

Modules Updated: Model, Preprocessor, VideoDatasetHandler
Script added: test_datasethandler.py

```shell
python test_datagen.py -m "FaceTrack_rPPG/DeepPhys" -id " list of input videos" -bs "batch_size"

```
Training script being added 