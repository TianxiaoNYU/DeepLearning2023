#!/bin/bash
#
#SBATCH --job-name=ML2022
#SBATCH -N 1
#SBATCH --cpus-per-task=16 # Request that ncpus be allocated per process.
#SBATCH -t 100:00 # Runtime in D-HH:MM
#SBATCH --output=./slurm/out/DL2023_image-%A_%a.out
#SBATCH --error=./slurm/error/DL2023_image-%A_%a.error
#SBATCH --array=0-472 # job array index

start=`date +%s`
module load anaconda3
conda activate /gpfs/data/ruggleslab/home/chenz05/course/ML2022/ml_env
module unload anaconda3 ## anaconda3 and python both have python, mute python form anacodna3.

jobs=($(tail -n +2 ./input/metadata/trimmed_mo_gdc_sample_sheet.tsv | grep "Copy Number Variation"  | cut -f 6))      
sampleID=${jobs[${SLURM_ARRAY_TASK_ID}]}
download_path_list=($(tail -n +2 ./input/metadata/trimmed_mo_gdc_sample_sheet.tsv  | grep "Copy Number Variation"  | cut -f 1))
download_path=${download_path_list[${SLURM_ARRAY_TASK_ID}]}
echo $sampleID $download_path
module load  gdc-client/1.6.0
gdc-client download $download_path --dir ./input/tmp
mv ./input/tmp/${download_path}  ./input/CopyNumVar/${sampleID}



jobs=($(tail -n +2 ./input/metadata/trimmed_mo_gdc_sample_sheet.tsv | grep "DNA Methylation"  | cut -f 6))      
sampleID=${jobs[${SLURM_ARRAY_TASK_ID}]}
download_path_list=($(tail -n +2 ./input/metadata/trimmed_mo_gdc_sample_sheet.tsv  | grep "DNA Methylation"  | cut -f 1))
download_path=${download_path_list[${SLURM_ARRAY_TASK_ID}]}
echo $sampleID $download_path
module load  gdc-client/1.6.0
gdc-client download $download_path --dir ./input/tmp
mv ./input/tmp/${download_path}  ./input/methylation/${sampleID}



jobs=($(tail -n +2 ./input/metadata/trimmed_mo_gdc_sample_sheet.tsv | grep "Transcriptome Profiling"  | cut -f 6))      
sampleID=${jobs[${SLURM_ARRAY_TASK_ID}]}
download_path_list=($(tail -n +2 ./input/metadata/trimmed_mo_gdc_sample_sheet.tsv  | grep "Transcriptome Profiling"  | cut -f 1))
download_path=${download_path_list[${SLURM_ARRAY_TASK_ID}]}
echo $sampleID $download_path
module load  gdc-client/1.6.0
gdc-client download $download_path --dir ./input/tmp
mv ./input/tmp/${download_path}  ./input/RNA/${sampleID}




