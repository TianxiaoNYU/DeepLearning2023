#!/bin/bash
#SBATCH --time=13:00:00
#SBATCH --mem=70GB
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/gpfs/home/chenz05/DL2023/output/3_coLearn/TZ/coopLearning_test_%j.out
#SBATCH --job-name=coopLearning_test

source /gpfs/home/tz2525/.bashrc
conda activate DL2023
python -u /gpfs/home/chenz05/DL2023/script/3_coLearn/TZ_coopLearning.py > /gpfs/home/chenz05/DL2023/output/3_coLearn/TZ/all_coopLearning_1_10-6_test.txt
