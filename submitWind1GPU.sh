#!/bin/sh
#BSUB -n 2
#BSUB -q g_medium
#BSUB -P R000
#BSUB -J SS_1gpu
#BSUB -o ./l_w.out
#BSUB -e ./l_w.err
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=400G]"
#python train.py
python train_wind.py
