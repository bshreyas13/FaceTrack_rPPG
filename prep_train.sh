#!/bin/bash

#SBACTH -J rPPG_training
#SBATCH -t 20:00:00
#SBATCH -N1
#SBACTH -p normal_q
#SBATCH -A ece-cs6524

#load module
module reset
module load Anaconda3/2020.11
module load TensorFlow/2.4.1-fosscuda-2020b

# Navigate to FaceTrack_rPPG
cd /home/bshreyas/rPPG/FaceTrack_rPPG 
source activate rPPG

echo "Running Job : Preparing Dataset and training" 

python3 _data.py -ds "../Roi_Videos" -lp "../Labels" -fo

exit;