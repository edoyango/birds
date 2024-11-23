#!/bin/bash
#SBATCH --job-name yolov11m_864_3-4_300_train
#SBATCH --output %x.out
#SBATCH --cpus-per-task 48
#SBATCH --mem 200G
#SBATCH -p gpuq
#SBATCH --gres gpu:A30:4
#SBATCH --qos bonus
#SBATCH --requeue
#SBATCH --time 12:0:0

set -ue
readarray -t -d _ args <<< ${SLURM_JOB_NAME}
model=${args[0]}
imgsz=${args[1]}
version=${args[2]}
epochs=${args[3]}

module load apptainer/1.3.3

CONTAINER=docker://ultralytics/ultralytics:8.3.1
MOUNTS="/vast,/stornext,/vast/scratch/users/yang.e/birds/runs:/ultralytics/runs,/vast/scratch/users/yang.e/datasets:/ultralytics/datasets"
export APPTAINER_TMPDIR=/dev/shm
export NO_ALBUMENTATIONS_UPDATE=1

apptainer exec -B "$MOUNTS" --nv $CONTAINER \
	yolo train cfg=/vast/scratch/users/yang.e/birds/cfg.yaml

cp ../datasets/birds-$version/data.yaml ../datasets/birds-$version/data-val.yaml
sed -i 's@val: valid/images@val: test/images@g' ../datasets/birds-$version/data-val.yaml
apptainer exec -B "$MOUNTS" --nv $CONTAINER \
	yolo detect val \
		data=../datasets/birds-$version/data-val.yaml \
		model=runs/detect/${SLURM_JOB_NAME}/weights/best.pt \
		name=${SLURM_JOB_NAME}_test \
		exist_ok=True \
		imgsz=$imgsz

sbatch --wrap "apptainer exec -B "$MOUNTS" --nv $CONTAINER yolo export model=runs/detect/${SLURM_JOB_NAME}/weights/best.pt format=engine batch=20 half=True" -c 4 --mem 20G --gres gpu:A30:1 -p gpuq
