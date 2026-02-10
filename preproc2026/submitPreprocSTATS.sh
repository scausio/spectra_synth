#!/bin/bash
#BSUB -J wam_spcs_reduce
#BSUB -P R000
#BSUB -q p_short
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -M 40G
#BSUB -o wam_stats_reduce.%J.out
#BSUB -e wam_stats_reduce.%J.err


# Prevent oversubscription (VERY IMPORTANT)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -----------------------------
# RUN SCRIPT
# -----------------------------
python preprocSTATS.py

