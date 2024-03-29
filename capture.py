#!/usr/bin/env python3

import cv2, time, os, stat
from ultralytics import YOLO
from datetime import datetime
import torch

DT_FORMAT_STR = "%Y-%m-%d_%H%M%S"

TIME_FORMAT_STR = "%H-%M-%S"
DATE_FORMAT_STR = "%Y-%m-%d"

def subclassify(detection_results, cls_model):

    tmp_box_data = detection_results.boxes.data.clone()

    res = cls_model(
        [
            detection_results[0].orig_img[box[1]:box[3], box[0]:box[2], :] for box in tmp_box_data.round().int()
        ],
        imgsz=64
    )

    if res[0].probs.top1conf.item() > 0.8:

        tmp_box_data[:, -1] = torch.Tensor([i.probs.top1+1 for i in res])
        detection_results.boxes.data = tmp_box_data
        species_names = res[0].names
        new_names = detection_results.names
        for k, v in species_names.items():
            new_names[k+1] = v
        detection_results.names = new_names

    return detection_results

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        prog="bird-detector.py",
        description="Detects birds from a given rtmp stream link and saves images of the brids detected."
    )
    parser.add_argument("-l", "--link", 
                        help="rtmp stream to pull from", 
                        default="rtmp://localhost/live/birbs"
                        )
    parser.add_argument("-d", "--detection-model",
                        help="Model to use for detection.",
                        default="yolov8s.pt"
                        )
    parser.add_argument("-c", "--classifier-model",
                        help="Model to use for sub-classification.",
                        default="yolov8s-cls.pt"
                        )

    args = parser.parse_args()

    model = YOLO(args.detection_model)
    cls_model = YOLO(args.classifier_model)

    vidcap = cv2.VideoCapture(args.link)

    if not vidcap.isOpened():
        raise RuntimeError("Failed to open stream")

    x = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    y = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for k, v in model.names.items():
        if v == "bird":
            bird_class = k

    last = 0
    suc = True
    while suc:


        if time.monotonic() - last > 10:
            suc, img = vidcap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            last = time.monotonic()
            img_cropped = img[24:, :].copy()
            results = model(source=img_cropped, classes=[bird_class], augment=True, conf=0.4, imgsz=(1056,1920))

            if bird_class in results[0].boxes.cls:

                # setup filenames
                now = datetime.now()
                today = now.strftime(DATE_FORMAT_STR)
                fsuffix = f"{now.strftime(TIME_FORMAT_STR)}.jpg"
                today_dir = os.path.join("observations", today)
                trigger_dir = os.path.join(today_dir, "triggers")
                trigger_img = os.path.join(trigger_dir, f"trigger_{fsuffix}")
                original_dir = os.path.join(today_dir, "originals")
                original_img = os.path.join(original_dir, f"original_{fsuffix}")
                
                # create directories
                os.makedirs(trigger_dir, exist_ok=True)
                os.makedirs(original_dir, exist_ok=True)

                # apply subclassifier to detection results
                subclassify(results[0], cls_model)

                # save results to disk
                results[0].save(trigger_img)
                cv2.imwrite(original_img, img)

                # making sure all files are word rwxable for docker
                os.chmod(today_dir, 0o777)
                os.chmod(trigger_dir, 0o777)
                os.chmod(original_dir, 0o777)
                os.chmod(trigger_img, 0o666)
                os.chmod(original_img, 0o666)

        else:
            suc = vidcap.grab()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    vidcap.release()
