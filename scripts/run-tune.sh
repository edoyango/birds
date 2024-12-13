#!/bin/bash
#SBATCH --job-name yolov11s-864-tune-alldata-cont
#SBATCH --output %x.out
#SBATCH --cpus-per-task 16
#SBATCH --mem 40G
#SBATCH -p gpuq
#SBATCH --gres gpu:1
#SBATCH --constraint Ampere
#SBATCH --prefer A100
#SBATCH --qos bonus
#SBATCH --requeue
##SBATCH --time 4-
##SBATCH --node

set -ue
version=$1
tunedir=$2
#date=$2
#imgsz=$3

module load apptainer/1.3.3

CONTAINER=oras://ghcr.io/edoyango/ultralytics:8.3.1 #docker://ultralytics/ultralytics:8.3.1
MOUNTS="/vast,/stornext,/vast/scratch/users/yang.e/datasets:/ultralytics/datasets,$(mkdir -p $tunedir && realpath $tunedir):/ultralytics/runs"
APPTAINER_TMPDIR=/dev/shm

apptainer exec -B "$MOUNTS" --nv $CONTAINER \
	python -c "
from ultralytics import YOLO
import yaml

model = YOLO('yolo11s.pt')
#model = YOLO('/vast/scratch/users/yang.e/birds/models/yolov11-birbs.pt')
#model = YOLO('/vast/scratch/users/yang.e/birds/runs/detect/yolo11s_704_3-7_120_48/weights/best.pt')

#space = {
#    'lr0': (1e-5, 1e-1),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
#    'lrf': (0.0001, 0.1),  # final OneCycleLR learning rate (lr0 * lrf)
#    'momentum': (0.7, 0.98, 0.3),  # SGD momentum/Adam beta1
#    'weight_decay': (0.0, 0.001),  # optimizer weight decay 5e-4
#    'warmup_epochs': (0.0, 5.0),  # warmup epochs (fractions ok)
#    'warmup_momentum': (0.0, 0.95),  # warmup initial momentum
#    'box': (1.0, 20.0),  # box loss gain
#    'cls': (0.2, 4.0),  # cls loss gain (scale with pixels)
#    'dfl': (0.4, 6.0),  # dfl loss gain
#}

model.tune(data='/vast/scratch/users/yang.e/datasets/birds-$version/data.yaml', device=0, iterations=300, imgsz=704, epochs=120, batch=48, optimizer='AdamW', cache=True, exist_ok=True, iou=0.5, augment=True, val=False, save=False)
"
