#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/gpfs/home/chenz05/DL2023/output/2_baseMulti/TZ/out_TZ_multiomics_%j.out
#SBATCH --job-name=Multi

source /gpfs/home/tz2525/.bashrc
conda activate DL2023
python -u /gpfs/home/chenz05/DL2023/script/1_other/TZ_multiomics_test.py > /gpfs/home/chenz05/DL2023/output/2_baseMulti/TZ/TZ_multiomics_testAUC_res_new_ROC.txt
