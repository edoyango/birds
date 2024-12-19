import os
import cv2
import sys
import argparse
import csv
import datetime
from pathlib import Path
import random

import numpy as np
import multiprocessing as mp

from rknn_yolov5 import RKNN_model, COCO_test_helper, draw


OBJ_THRESH = 0.45
NMS_THRESH = 0.5

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (704, 704)  # (width, height), such as (1280, 736)

CLASSES = ("Blackbird", "Butcherbird", "Currawong", "Dove", "Lorikeet", "Myna", "Sparrow", "Starling", "Wattlebird")

FFMPEG_CMD = "ffmpeg -y -hide_banner -loglevel error -i {input_video} -init_hw_device rkmpp=hw -filter_hw_device hw -vf hwupload,scale_rkrga=w=864:h=486 -c:v hevc_rkmpp -qp_init 20 {output_video}"

def open_video(vname):

    cap = cv2.VideoCapture(vname)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if (cap.isOpened() == False):  
        raise RuntimeError("Error reading video file")

    return cap

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

def count_detections(classes):

    counts = {}
    for i in classes:
        ii = int(i)
        if counts.get(ii):
            counts[ii] += 1
        else:
            counts[ii] = 1
    
    return {CLASSES[k]: v for k, v in counts.items()}

class detected_bird_video:
    def __init__(self, output_path, fps, width, height, minframes, prefix):

        # define video name
        # assumes there is <1s difference between when the frame is captured
        # and video is opened by worker
        self.vid_name = f"{prefix}{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        # define and create output directories
        self.trigger_dir = Path(output_path) / "triggers"
        self.trigger_firstframes_dir = self.trigger_dir / "first_frames"
        self.original_dir = Path(output_path) / "originals"
        self.original_firstframes_dir = self.original_dir / "first_frames"
        self.trigger_firstframes_dir.mkdir(parents=True, exist_ok=True)
        self.original_firstframes_dir.mkdir(parents=True, exist_ok=True)

        # save video properties
        self.fps = fps
        self.width = width
        self.height = height
        self.minframes = minframes
        
        # paths
        self.trigger_vid_path = self.trigger_dir / f"trigger-{self.vid_name}.mp4"
        self.original_vid_path = self.original_dir / f"original-{self.vid_name}.mp4"
        self.trigger_firstframe_path = self.trigger_firstframes_dir / f"trigger-{self.vid_name}.jpg"
        self.original_firstframe_path = self.original_firstframes_dir / f"original-{self.vid_name}.jpg"
        self.meta_csv = Path(output_path) / "meta.csv" # stores all video metadata

        # open video
        self.cap_trigger = cv2.VideoWriter(
            self.trigger_vid_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height)
        )
        self.cap_original = cv2.VideoWriter(
            self.original_vid_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height)
        )

        # initialise output video metadata
        self.nframes = 0
        self.total_instances = 0
        self.total_class_count = {c: 0 for c in CLASSES}
        self.opened = self.cap_trigger.isOpened() and self.cap_original.isOpened()
        if not self.opened:
            raise RuntimeError("Problem opening trigger or original video")
    
    def write(self, trigger_frame, original_frame, classes):
        """
        Write frames to trigger and original videos and record number of classes detected.
        """

        if self.nframes == 0:
            cv2.imwrite(self.trigger_firstframe_path, trigger_frame)
            cv2.imwrite(self.original_firstframe_path, original_frame)
        
        self.cap_trigger.write(trigger_frame)
        self.cap_original.write(original_frame)

        self.nframes += 1

        classes_count = {} if classes is None else count_detections(classes)
        for k, v in classes_count.items():
            self.total_class_count[k] += v
            self.total_instances += v
    
    def release(self, p_keep=0.9):
        """
        Release output video captures, add video metadata to meta.csv, and delete output videos if too short.
        """

        # initialise row and header
        row = [
            "N/A",
            "N/A",
            self.original_vid_path,
            self.original_firstframe_path,
            self.trigger_vid_path,
            self.trigger_firstframe_path,
            self.nframes,
            self.total_instances/self.nframes
        ]
        header = ["reference video path", 
                  "start time", 
                  "original video path", 
                  "original first frame path", 
                  "trigger video path", 
                  "trigger first frame path", 
                  "nframes", 
                  "average ninstance per frame"
                 ]
        
        # add class names and data to header/row
        for k, v in self.total_class_count.items():
            header.append(k)
            row.append(str(v))

        # append to meta.csv if it exists, otherwise create a new one with headers
        if self.meta_csv.exists() and self.meta_csv.is_file():
            with self.meta_csv.open(mode="a") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(row)
        else:
            with self.meta_csv.open(mode="w") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(header)
                csvwriter.writerow(row)
        
        # close caps
        self.cap_trigger.release()
        self.cap_original.release()

        # compress output videos, removing if they don't meet the minframes threshold
        if self.nframes < self.minframes:
            self.trigger_vid_path.unlink(missing_ok=True)
            self.original_vid_path.unlink(missing_ok=True)
        else:
            # randomly keep fullres video for training
            if random.random() > p_keep:
                err = os.system(
                    f"""{FFMPEG_CMD.format(input_video=self.trigger_vid_path, output_video=self.trigger_vid_path.parent / (self.trigger_vid_path.stem+"-compressed"+self.trigger_vid_path.suffix))} && rm {self.trigger_vid_path}
                        {FFMPEG_CMD.format(input_video=self.original_vid_path, output_video=self.original_vid_path.parent / (self.original_vid_path.stem+"-compressed"+self.original_vid_path.suffix))} && rm {self.original_vid_path}
                    """
                )
            else:
                err = os.system(
                    f"""{FFMPEG_CMD.format(input_video=self.trigger_vid_path, output_video=self.trigger_vid_path.parent / (self.trigger_vid_path.stem+"-compressed"+self.trigger_vid_path.suffix))}
                        {FFMPEG_CMD.format(input_video=self.original_vid_path, output_video=self.original_vid_path.parent / (self.original_vid_path.stem+"-compressed"+self.original_vid_path.suffix))}
                    """
                )
            assert err==0, "Video compression failed."
            

        # tag as closed
        self.opened = False
    
    def isOpened(self):
        """
        Helper function to indicate if videos are open.
        """
        return self.opened

def video_writer_worker(queue, w, h, wait_limit, output_path):

    # initialize output video
    output_video = None

    # start reading frames from master
    wait_counter = sys.maxsize
    while True:
        res = queue.get()
        if res is None:
            break
        else:
            bird_detected = res["classes"] is not None
            wait_counter = 0 if bird_detected else wait_counter + 1
            if wait_counter < wait_limit:
                if not output_video or not output_video.isOpened():
                    output_video = detected_bird_video(output_path, 10, w, h, 50, "")
                output_video.write(res["drawn image"], res["original image"], res["classes"])
            elif output_video and output_video.isOpened():
                output_video.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect birds from a video feed.')
    # basic params
    parser.add_argument('--model_path', type=str, required= True, help='model path, could be .pt or .rknn file')
    parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')

    # data params
    parser.add_argument('--video', '-v', type=str, default='0', help='Video to watch. Can be either video device index, e.g. 0, or video file e.g. video.mkv.')
    parser.add_argument('--anchors', type=str, default='../model/anchors_yolov5.txt', help='target to anchor file, only yolov5, yolov7 need this param')

    # output
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("."), help="Output directory to save inference results.")
    parser.add_argument("--max-frames", "-m", type=int, default=0, help="Maximum number of frames to record for.")

    args = parser.parse_args()

    # load anchors
    with open(args.anchors, 'r') as f:
        values = [float(_v) for _v in f.readlines()]
        anchors = np.array(values).reshape(3,-1,2).tolist()
    print("use anchors from '{}', which is {}".format(args.anchors, anchors))
    
    # init model
    model = RKNN_model(args.model_path, args.target, args.device_id)

    # create letterboxing class
    co_helper = COCO_test_helper()

    # open video and define some metadata
    cap = open_video(args.video)
    fps=10.0 # temporary fix
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames == 0: args.max_frames = sys.maxsize
    total_frames = args.max_frames if total_frames < 0 else min(total_frames, args.max_frames)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wait_limit = 50
    print(f"""INPUT VIDEO: {args.video}
    Resolution: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}x{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}
    FPS:        {fps}
OUTPUT DIRECTORY: result/{{originals,triggers}}
DETECTION MODEL: {args.model_path}
WAIT LIMIT: {wait_limit} frames
MAX FRAMES: {args.max_frames} frames

Beginning processing...
""")
    
    # initiate video writer
    frame_queue = mp.Queue()
    worker = mp.Process(
        target=video_writer_worker,
        args=(frame_queue, w, h, wait_limit, args.output_dir)
    )
    worker.start()

    # begin reading
    suc = True
    iframe = 0
    while suc and iframe < total_frames:

        # read frame from input feed
        suc, frame = cap.read()

        if not suc:
            raise("Error reading frame from input feed.")

        # letterbox video as per training
        pad_color = (0,0,0)
        img = co_helper.letter_box(im= frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # inference
        boxes, classes, scores = model.infer([img], anchors, IMG_SIZE, NMS_THRESH)

        # print number of detections to terminal
        print(f"frame {iframe} {0 if classes is None else len(classes)} birds", end="\r")

        # draw boxes on top of original frame
        img_p = frame.copy()
        if boxes is not None:
            draw(img_p, co_helper.get_real_box(boxes), scores, classes, CLASSES)

        # check video writer worker is still alive
        if worker.exitcode:
            worker.close()
            raise RuntimeError(f"Worker has died with error code {worker.exitcode}")
        
        # send frame and classes to video worker
        frame_queue.put(
            {"classes": classes.copy() if classes is not None else None,
             "drawn image": img_p.copy(),
             "original image": frame.copy(),
            }
        )

        iframe +=1
    
    # send signal to end video writeing
    frame_queue.put(None)

    # release model and capture
    model.release()
    cap.release()