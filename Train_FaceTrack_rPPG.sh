#!/bin/bash

#SBACTH -J rPPG_training
#SBATCH -t 20:00:00
#SBATCH -N6
#SBACTH -p t4_normal_q
#SBATCH -A ece-cs6524

#load module
module reset
module load Anaconda3/2020.11
module load TensorFlow/2.4.1-fosscuda-2020b

# Navigate to FaceTrack_rPPG
cd /home/bshreyas/rPPG/FaceTrack_rPPG 
source activate rPPG

echo "Running Job : Preparing Dataset and training FaceTrack_rPPG" 

python3 build_train_model_v2.py -model "FaceTrack_rPPG" -ap "../Dataset/Roi" -mp "../Dataset/Nd" -lp "../Labels" -wtxt -wtfr -rmtxt -rmtfr

exit;