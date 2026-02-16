#!/bin/sh
#BSUB -n 1
##BSUB -q p_short
#BSUB -q g_short
#BSUB -gpu "num=1"
#BSUB -P 0710
#BSUB -J valid
#BSUB -R "rusage[mem=80G]"
#BSUB -e valid.e
#BSUB -o valid.o


python inference_and_compare.py \
  --model_path /work/cmcc/sc33616/work/spectra_synth/output_unet_msle/best_model.pt \
  --stats_dir /work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026/stats_grid \
  --spc_dir /work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026/spcs_grid \
  --depth /work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026/stats_grid/dpt.nc \
  --scaler /work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026/scalers/scalers.json \
    --batch_size 128 \
    --num_samples 200  --compute_yamaguchi
