#!/bin/bash
#SBATCH --job-name yolo-train
#SBATCH --output %x.out
#SBATCH --cpus-per-task 48
#SBATCH --mem 200G
#SBATCH -p gpuq
#SBATCH --gres gpu:A30:4
#SBATCH --qos bonus

set -ue
version=$1
date=$2
imgsz=$3

module load apptainer/1.2.3

apptainer exec -B /vast -B /stornext --nv docker://ultralytics/ultralytics:8.2.18 \
	yolo train \
		data=../datasets/birds-$version/data.yaml \
		model=models/yolov9-birbs.pt \
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
apptainer exec -B /vast -B /stornext --nv docker://ultralytics/ultralytics:8.2.18 \
	yolo detect val \
		data=../datasets/birds-$version/data-val.yaml \
		model=runs/detect/birds-$version-$date-$imgsz-train/weights/best.pt \
		name=birds-$version-$date-$imgsz-test \
		exist_ok=True \
		imgsz=$imgsz
