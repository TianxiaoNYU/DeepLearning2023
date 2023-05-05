#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/gpfs/home/chenz05/DL2023/output/2_baseMulti/out_TZ_multiomics_%j.out
#SBATCH --job-name=Multi

source /gpfs/home/tz2525/.bashrc
conda activate DL2023
python -u /gpfs/home/chenz05/DL2023/script/2_baseMulti/TZ_step_4c_resnet50_subset.py > /gpfs/home/chenz05/DL2023/output/2_baseMulti/TZ_multiomics_res.txt
