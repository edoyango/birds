#!/bin/bash -l

#SBATCH --mail-user edward_yang_125@hotmail.com
#SBATCH --mail-type ALL
#SBATCH --cpus-per-task 4
#SBATCH --mem 8GB
#SBATCH --job-name daily-processing
#SBATCH --output /vast/scratch/users/yang.e/tmp/%x-%j.o
#SBATCH --error /vast/scratch/users/yang.e/tmp/%x-%j.e
#SBATCH --chdir /vast/scratch/users/yang.e/birds/nf

set -eu

module load nextflow

printf -v date '%(%Y-%m-%d)T' -1

 GMAIL_APP_PWD=bpcshpyjugjpmbvy nextflow run main.nf --date $date -profile wehi --model_detect ../models/yolov9-birbs.engine --model_cls ../models/yolov8-cls-birbs.pt --outdir test-output --email_list ../email-lists.csv -process.errorStrategy retry -resume

nextflow clean -f
