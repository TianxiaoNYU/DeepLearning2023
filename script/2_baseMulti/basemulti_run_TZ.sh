#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu4_dev
#SBATCH --output=/gpfs/home/chenz05/DL2023/output/2_baseMulti/TZ/basemulti_%j.out
#SBATCH --job-name=resnet_normalized

source /gpfs/home/tz2525/.bashrc
conda activate DL2023
python -u /gpfs/home/chenz05/DL2023/script/2_baseMulti/TZ_basemulti_test.py > /gpfs/home/chenz05/DL2023/output/2_baseMulti/TZ/basemulti_testing_res_wholedata.txt
