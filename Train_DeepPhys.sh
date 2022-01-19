#!/bin/bash

#SBACTH -J rPPG_training
#SBATCH -t 80:00:00
#SBATCH -N4
#SBACTH -p t4_normal_q
#SBATCH -A ece-cs6524

#load module
module reset
module load Anaconda3/2020.11
module load TensorFlow/2.4.1-fosscuda-2020b

# Navigate to FaceTrack_rPPG
cd /home/bshreyas/rPPG/FaceTrack_rPPG 
source activate rPPG

echo "Running Job : Preparing Dataset and training DeepPhys" 

python3 build_train_model_v2.py -m "DeepPhys" -ap "../Dataset/Roi" -mp "../Dataset/Nd" -lp "../Scaled_Labels" -tfr_path "../Dataset/TFRecords/DeepPhys_SL" -bs "10" -ep "20" -sbst "0.5"

exit;