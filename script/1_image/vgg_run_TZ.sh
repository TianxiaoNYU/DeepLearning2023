#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/gpfs/home/chenz05/DL2023/output/1_image/TZ/vgg_%j.out
#SBATCH --job-name=VGG_normalized

source /gpfs/home/tz2525/.bashrc
conda activate DL2023
python -u /gpfs/home/chenz05/DL2023/script/1_image/TZ_vgg.py > /gpfs/home/chenz05/DL2023/output/1_image/TZ/vgg_res_normalized_wholedata.txt
