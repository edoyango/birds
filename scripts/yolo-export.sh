#!/bin/bash
#SBATCH --time 60 -c 16 --mem 100G --gres gpu:A30:1 -p gpuq 
module purge
module load apptainer/1.2.3
apptainer exec -B /vast,/stornext --nv docker://ultralytics/ultralytics:8.2.42 yolo export model=../models/yolov10-birbs.pt format=engine batch=20 half=True
