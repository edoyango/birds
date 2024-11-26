#!/bin/bash -l

#SBATCH --mail-user edward_yang_125@hotmail.com
#SBATCH --mail-type ALL
#SBATCH --cpus-per-task 4
#SBATCH --mem 8GB
#SBATCH --job-name daily-processing
#SBATCH --output /vast/scratch/users/yang.e/tmp/%x-%j.o
#SBATCH --error /vast/scratch/users/yang.e/tmp/%x-%j.e
#SBATCH --chdir /vast/scratch/users/yang.e/birds/nf
#SBATCH --time 360

set -eu

module load nextflow/24.04.2

printf -v date '%(%Y-%m-%d)T' -1

export APPTAINER_TMPDIR=/vast/scratch/users/yang.e/.stmp
export APPTAINER_CACHEDIR=/vast/scratch/users/yang.e/.scache
export SINGULARITY_TMPDIR=$APPTAINER_TMPDIR SINGULARITY_CACHEDIR=$APPTAINER_CACHEDIR

 GMAIL_APP_PWD=bpcshpyjugjpmbvy nextflow run main.nf --date $date -profile wehi --model_detect ../models/yolov11-birbs.engine --outdir test-output --email_list ../email-lists.csv -process.errorStrategy retry -resume --nsamples 4 --conf 0.7

nextflow clean -f
