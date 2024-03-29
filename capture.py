#!/usr/bin/env python3

import cv2, time, os, stat
from ultralytics import YOLO
from datetime import datetime

DT_FORMAT_STR = "%Y-%m-%d_%H%M%S"

TIME_FORMAT_STR = "%H-%M-%S"
DATE_FORMAT_STR = "%Y-%m-%d"

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

    args = parser.parse_args()

    model = YOLO("yolov8l.pt")

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
            img_cropped = img[376:1080, :].copy()
            results = model(source=img_cropped, classes=[bird_class], augment=True, conf=0.4, imgsz=(y-376, x))

            if bird_class in results[0].boxes.cls:

                now = datetime.now()
                today = now.strftime(DATE_FORMAT_STR)
                fsuffix = f"{now.strftime(TIME_FORMAT_STR)}.jpg"
                today_dir = os.path.join("observations", today)
                trigger_dir = os.path.join(today_dir, "triggers")
                trigger_img = os.path.join(trigger_dir, f"trigger_{fsuffix}")
                original_dir = os.path.join(today_dir, "originals")
                original_img = os.path.join(original_dir, f"original_{fsuffix}")
                
                os.makedirs(trigger_dir, exist_ok=True)
                os.makedirs(original_dir, exist_ok=True)
                results[0].save(trigger_img)
                cv2.imwrite(original_img, img)

                # making sure all files are word rwxable
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
