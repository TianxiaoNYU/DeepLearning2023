#!/bin/bash
#
#SBATCH --job-name=ML2022
#SBATCH -N 1
#SBATCH --cpus-per-task=16 # Request that ncpus be allocated per process.
#SBATCH -t 100:00 # Runtime in D-HH:MM
#SBATCH --output=./slurm/out/DL2023_image-%A_%a.out
#SBATCH --error=./slurm/error/DL2023_image-%A_%a.error
#SBATCH --array=0-473 # job array index


image_num=$1
image_size=$2

start=`date +%s`
module load anaconda3
conda activate /gpfs/data/ruggleslab/home/chenz05/course/ML2022/ml_env
module unload anaconda3 ## anaconda3 and python both have python, mute python form anacodna3.

jobs=($(tail -n +2 ./input/metadata/trimed_gdc_sample_sheet.tsv | grep "Biospecimen"  | cut -f 8)) 
sampleID=${jobs[${SLURM_ARRAY_TASK_ID}]}

download_path_list=($(tail -n +2 ./input/metadata/trimed_gdc_sample_sheet.tsv | grep "Biospecimen"  | cut -f 2))
download_path=${download_path_list[${SLURM_ARRAY_TASK_ID}]}

echo $sampleID $download_path

###########################################################################
###########################################################################
###                                                                     ###
###                     STEP 1 download whole SVS file                  ###
###                                                                     ###
###########################################################################
###########################################################################
module load  gdc-client/1.6.0 
gdc-client download $download_path --dir ./input/tmp
mv ./input/tmp/${download_path}  ./input/tmp/${sampleID}

echo -e "\n\n\n\n" $sampleID "finished downloading" "\n\n\n\n"

###########################################################################
###########################################################################
###                                                                     ###
###                     STEP 2 subsetting image                         ###
###                                                                     ###
###########################################################################
###########################################################################


python -W ignore ./script/1_image/split_whole.image.py ./input/tmp/${sampleID} ./input/image/${sampleID}  $image_num $image_size

echo -e "\n\n\n\n" $sampleID "finished splitting" "\n\n\n\n"

###have to remove picture due to space limitation.
rm -r ./input/tmp/${sampleID} 
 





end=`date +%s`
runtime=$((end-start)) 
echo  "this process was finished in" $runtime  "seconds"
