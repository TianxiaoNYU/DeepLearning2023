#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/gpfs/home/chenz05/DL2023/output/1_image/TZ/resnet50_%j.out
#SBATCH --job-name=Resnet

source /gpfs/home/tz2525/.bashrc
conda activate DL2023
python -u /gpfs/home/chenz05/DL2023/script/1_image/TZ_resnet50.py > /gpfs/home/chenz05/DL2023/output/1_image/TZ/resnet50_res.txt
