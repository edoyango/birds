import os, torch, cv2, numpy as np, time
from ultralytics import YOLO
import torch
from datetime import datetime

DT_FORMAT_STR = "%Y-%m-%d_%H%M"

def capture(video_capture, file_name: str = "output.mp4", duration: float = 60) -> None:
    """
    Function to capture webcam video for a set duration
    """

    # create video writer opbject
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    size = (
        int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
        int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    out = cv2.VideoWriter(file_name, 
        fourcc, 
        video_capture.get(cv2.CAP_PROP_FPS), 
        size
    )

    start = time.monotonic()

    success = True
    while time.monotonic() - start < duration and success:

        success, frame = video_capture.read()

        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()



if __name__ == "__main__":

    model = YOLO("yolov8l.pt")

    vidcap = cv2.VideoCapture(0)

    vidcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    vidcap.set(cv2.CAP_PROP_FPS, 20)
    vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # capture(vidcap, duration=10)

    for k, v in model.names.items():
        if v == "bird":
            bird_class = k

    suc = True
    while suc:

        suc, img = vidcap.read()

        img = img[:800, :1280]

        results = model(source=img, classes=[bird_class], imgsz=(1056, 1920), augment=True, conf=0.5)

        if bird_class in results[0].boxes.cls:
            now = datetime.now().strftime(DT_FORMAT_STR)
            results[0].save(f"trigger{now}.jpg")
            capture(vidcap, f"output{now}.mp4", 60)
            time.sleep(60)


    vidcap.release()
