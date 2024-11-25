#!/bin/bash
#SBATCH --job-name yolov11s_train
#SBATCH --output %x.out
#SBATCH --cpus-per-task 8
#SBATCH --mem 24G
#SBATCH -p gpuq
#SBATCH --gres gpu:A30:1
##SBATCH --qos bonus
#SBATCH --requeue
#SBATCH --time 12:0:0

set -ue
cfg_file=/vast/scratch/users/yang.e/birds/cfg.yaml
run_name="$(grep name $cfg_file | cut -d' ' -f 2 | tr -d '"')"
readarray -t -d _ args <<< "$run_name"
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
	yolo train cfg=$cfg_file

cp ../datasets/birds-$version/data.yaml ../datasets/birds-$version/data-val.yaml
sed -i 's@val: valid/images@val: test/images@g' ../datasets/birds-$version/data-val.yaml
apptainer exec -B "$MOUNTS" --nv $CONTAINER \
	yolo detect val \
		data=../datasets/birds-$version/data-val.yaml \
		model=runs/detect/${run_name}/weights/best.pt \
		name=${run_name}_test \
		exist_ok=True \
		imgsz=$imgsz

apptainer exec -B "$MOUNTS" --nv $CONTAINER yolo export model=runs/detect/${run_name}/weights/best.pt format=engine batch=50 half=True
