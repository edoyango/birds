Run detection training with 

```bash
yolo train data=${DATA_YML} model=yolov8${MODEL_SIZE}.pt pretrained=True epochs=3000 imgsz=864 cache=True batch=-1 pretrained=True workers=8 single_cls=True name=${JOB_NAME}-train patience=100 box=7.0
```

Run classification training with

```bash
yolo classify train data=birds-training model=yolov8${MODEL_SIZE}-cls.pt epochs=1500 imgsz=64 cache=True batch=-1 pretrained=True workers=8 name=${JOB_NAME}-train
```
