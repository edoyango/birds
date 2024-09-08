#!/bin/bash
#SBATCH --job-name yolo-train
#SBATCH --output %x.out
#SBATCH --cpus-per-task 48
#SBATCH --mem 200G
#SBATCH -p gpuq
#SBATCH --gres gpu:A30:4
#SBATCH --qos bonus
#SBATCH --requeue

set -ue
version=$1
date=$2
imgsz=$3

module load apptainer/1.3.3

CONTAINER=docker://ultralytics/ultralytics:8.2.79
MOUNTS="/vast,/stornext,/vast/scratch/users/yang.e/birds/runs:/ultralytics/runs,/vast/scratch/users/yang.e/datasets:/ultralytics/datasets"

apptainer exec -B "$MOUNTS" --nv $CONTAINER \
	yolo train \
		data=../datasets/birds-$version/data.yaml \
		model=yolov10b.pt \
		pretrained=True \
		epochs=300 \
		imgsz=$imgsz \
		cache=True \
		batch=64 \
		workers=48 \
		single_cls=True \
		name=birds-$version-$date-$imgsz-train \
		exist_ok=True \
		box=7.0 \
		device=0,1,2,3 \
		patience=0 \
		augment=False

cp ../datasets/birds-$version/data.yaml ../datasets/birds-$version/data-val.yaml
sed -i 's@val: valid/images@val: test/images@g' ../datasets/birds-$version/data-val.yaml
apptainer exec -B "$MOUNTS" --nv $CONTAINER \
	yolo detect val \
		data=../datasets/birds-$version/data-val.yaml \
		model=runs/detect/birds-$version-$date-$imgsz-train/weights/best.pt \
		name=birds-$version-$date-$imgsz-test \
		exist_ok=True \
		imgsz=$imgsz

sbatch --wrap "apptainer exec -B "$MOUNTS" --nv $CONTAINER yolo export model=runs/detect/birds-$version-$date-$imgsz-train/weights/best.pt format=engine batch=20 half=True" -c 4 --mem 20G --gres gpu:A30:1 -p gpuq
