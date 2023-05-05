#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --mem=70GB
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/gpfs/home/chenz05/DL2023/output/3_coLearn/whole/coopLearning_%j_pl.out
#SBATCH --job-name=coopLearning

source /gpfs/home/tz2525/.bashrc
conda activate DL2023
python -u /gpfs/home/chenz05/DL2023/script/3_coLearn/step_1c_resnet50_printloss.py > /gpfs/home/chenz05/DL2023/output/3_coLearn/whole/whole_coopLearning_pl.txt
