import os, torch, cv2, numpy as np, time
from ultralytics import YOLO
import torch
from datetime import datetime

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

    model = YOLO("yolov8m.pt")

    vidcap = cv2.VideoCapture(0)

    vidcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    vidcap.set(cv2.CAP_PROP_FPS, 20)
    vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # capture(vidcap, duration=10)

    for k, v in model.names.items():
        if v == "bird":
            bird_class = k

    suc = True
    while suc:

        suc, img = vidcap.read()

        results = model(source=img, classes=[bird_class], imgsz=(480, 640))

        if bird_class in results[0].boxes.cls:
            capture(vidcap, f"output{datetime.now()}.mp4", 60)
            time.sleep(60)


    vidcap.release()
