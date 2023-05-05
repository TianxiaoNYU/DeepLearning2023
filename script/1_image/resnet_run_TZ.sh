#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu4_dev
#SBATCH --output=/gpfs/home/chenz05/DL2023/output/1_image/TZ/resnet_%j_softmax.out
#SBATCH --job-name=resnet_normalized

source /gpfs/home/tz2525/.bashrc
conda activate DL2023
python -u /gpfs/home/chenz05/DL2023/script/1_image/TZ_resnet_test.py > /gpfs/home/chenz05/DL2023/output/1_image/TZ/resnet_testing_res_wholedata_softmax.txt
