#!/bin/sh
#BSUB -n 1
#BSUB -q p_short
#BSUB -P R000
#BSUB -J SS_1cpu
#BSUB -R "rusage[mem=20G]"

python run_experiments.py --experiment unet_deep
