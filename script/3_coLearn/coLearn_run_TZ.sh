#!/bin/bash
#SBATCH --time=30:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/gpfs/home/chenz05/DL2023/output/3_coLearn/test_coLearn_%j.out
#SBATCH --job-name=resnet_normalized

source /gpfs/home/tz2525/.bashrc
conda activate DL2023
python -u /gpfs/home/chenz05/DL2023/script/3_coLearn/TZ_coLearn_test.py > /gpfs/home/chenz05/DL2023/output/3_coLearn/testing_res_coLearn_0.3_1e-6.txt
