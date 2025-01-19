#!/bin/bash

set -eu

module load apptainer/1.3.3

version=$1
nfiles=$2

apptainer exec -B /vast -B /stornext oras://ghcr.io/edoyango/roboflow:latest python -c '
from roboflow import Roboflow
rf = Roboflow(api_key="0VMHB40ZqbuHhGf4KctL")
#project = rf.workspace("birds-0awev").project("birds-2-tvt0k")
project = rf.workspace("birds-0awev").project("birds-3-9jhp2")
version = project.version('$version')
dataset = version.download("yolov8")
'

srun -c 56 --mem 100G apptainer exec -B /vast -B /stornext docker://ultralytics/ultralytics:8.2.79 python3 scripts/augment-data.py -i -n 56 birds-3-$version $nfiles

sed -i "s@birds-3-$version/@@g" birds-3-$version/data.yaml
mv birds-3-$version ../datasets
