#!/bin/sh
#BSUB -n 4
#BSUB -q g_medium
#BSUB -P R000
#BSUB -J SS
#BSUB -o ./l_model.out
#BSUB -e ./l_model.err
#BSUB -gpu "num=2"
##BSUB -R "rusage[mem=300G]"
#BSUB -R "span[ptile=2]" #number of processes per nodes
#python train.py

export I_MPI_PLATFORM="skx"
export I_MPI_EXTRA_FILE_SYSTEM=1
export I_MPI_HYDRA_BOOTSTRAP="lsf"
export I_MPI_HYDRA_BRANCH_COUNT=$(( $( echo "${LSB_MCPU_HOSTS}" | wc -w ) / 2 ))
export I_MPI_HYDRA_COLLECTIVE_LAUNCH=1
mpirun -np 4 python train2gpus.py   #testmpi.py
