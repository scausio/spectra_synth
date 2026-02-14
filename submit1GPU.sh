#!/bin/sh
#BSUB -n 2
#BSUB -q g_medium
#BSUB -P R000
#BSUB -J SS_1gpu
#BSUB -o ./l_model.out
#BSUB -e ./l_model.err
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=50G]"
#python train.py
#python run_experiments.py --experiment hybrid_balanced
#python run_experiments.py --experiment unet_deep
python run_experiments.py --experiment unet_msle #unet_sparse_optimized
#train.py
