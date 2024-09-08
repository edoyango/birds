#!/bin/bash
#SBATCH --time 60 -c 16 --mem 100G --gres gpu:A30:1 -p gpuq --job-name yolo-export --output %x.out
module purge
module load apptainer/1.3.3
export APPTAINER_TMPDIR=/dev/shm
apptainer exec -B /vast,/stornext --nv docker://ultralytics/ultralytics:8.2.79 yolo export model=../models/yolov10-birbs.pt format=engine batch=20 half=True
rm -rf /dev/shm/*
