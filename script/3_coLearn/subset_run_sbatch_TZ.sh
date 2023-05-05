#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=70GB
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/gpfs/home/chenz05/DL2023/output/3_coLearn/ZCcoopLearning_%j.out
#SBATCH --job-name=coopLearning

source /gpfs/home/tz2525/.bashrc
conda activate DL2023
python -u /gpfs/home/chenz05/DL2023/script/3_coLearn/step_1c_resnet50_subset.py > /gpfs/home/chenz05/DL2023/output/3_coLearn/subset_ZC_coopLearning.txt
