#!/usr/bin/env python3

import cv2
import numpy as  np
import os
import datetime
import sys
import torch
import copy
import ultralytics
import pathlib
import csv
from collections import Counter

def open_video(vname):

    cap = cv2.VideoCapture(vname)

    if (cap.isOpened() == False):  
        raise RuntimeError("Error reading video file")

    return cap

def add_seconds_to_timestring(timestring: str, seconds: int):
    time_obj = datetime.datetime.strptime(timestring, "%H-%M-%S")
    time_diff = datetime.timedelta(seconds=seconds)
    return (time_obj + time_diff).strftime("%H-%M-%S")

def count_detections(detection_result):

    counts = {}
    for i in detection_result.boxes.cls.cpu().int():
        ii = int(i)
        if counts.get(ii):
            counts[ii] += 1
        else:
            counts[ii] = 1
    
    return {detection_result.names[k]: v for k, v in counts.items()}

class detected_birb_vid:

    def __init__(self, reference_cap, reference_vidname, frame_offset, outdir, minframes, instances_names):

        # check that reference_cap is open
        if not reference_cap.isOpened():
            raise RuntimeError("Input reference video capture not opened!")
        
        # save reference cap meta data
        self.fps = float(reference_cap.get(cv2.CAP_PROP_FPS))
        self.width = int(reference_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(reference_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.minframes = minframes

        # create paths for output original/trigger subvideos/images
        rv = pathlib.Path(reference_vidname)
        self.reference_vidpath = rv
        outpath = pathlib.Path(outdir)
        if not rv.is_file():
            raise RuntimeError("Input reference video path is not a file!")
        reference_timestr = rv.stem[-8:]
        self.prefix = rv.stem[:-8]
        reference_time = datetime.datetime.strptime(reference_timestr, "%H-%M-%S")
        self.start_time = reference_time + datetime.timedelta(seconds=frame_offset/self.fps)
        self.start_timestr = self.start_time.strftime("%H-%M-%S")
        self.original_vidpath = outpath.joinpath("originals", f"original-{self.prefix}{self.start_timestr}{rv.suffix}")
        self.trigger_vidpath = outpath.joinpath("triggers", f"trigger-{self.prefix}{self.start_timestr}{rv.suffix}")
        self.original_firstframepath = outpath.joinpath("originals", "first_frames", f"original-{self.prefix}{self.start_timestr}.jpg")
        self.trigger_firstframepath = outpath.joinpath("triggers", "first_frames", f"trigger-{self.prefix}{self.start_timestr}.jpg")
        self.meta_csvpath = outpath.joinpath("meta.csv")

        # open output subvideos
        self.original_cap = cv2.VideoWriter(
            str(self.original_vidpath),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height)
        )
        if not self.original_cap.isOpened(): raise RuntimeError("Couldn't create original video file")
        self.trigger_cap = cv2.VideoWriter(
            str(self.trigger_vidpath),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height)
        )
        if not self.original_cap.isOpened(): raise RuntimeError("Couldn't create trigger video file")

        # video metadata
        self.nframes = 0
        self.ninstances = 0
        self.ninstances_each = {n: 0 for n in instances_names}
        self.opened = True

    def write(self, original_frame, trigger_frame, instances):

        # save frame as image in case no frames have been written yet
        if not self.nframes:
            cv2.imwrite(str(self.original_firstframepath), original_frame)
            cv2.imwrite(str(self.trigger_firstframepath), trigger_frame)
        
        self.original_cap.write(original_frame)
        self.trigger_cap.write(trigger_frame)

        self.nframes += 1

        for k, v in instances.items():
            self.ninstances_each[k] += v
            self.ninstances += v
    
    def release(self):
        # release caps
        self.original_cap.release()
        self.trigger_cap.release()

        # delete video if too short
        if self.nframes < self.minframes:
            os.remove(str(self.original_vidpath))
            os.remove(str(self.trigger_vidpath))

        self.opened = False

        # write to metadatacsv
        row = [
            str(self.reference_vidpath),
            self.start_timestr,
            str(self.original_vidpath),
            str(self.original_firstframepath),
            str(self.trigger_vidpath),
            str(self.trigger_firstframepath),
            str(self.nframes),
            str(self.ninstances/self.nframes)
        ]
        header = ["reference video path", "start time", "original video path", "original first frame path", "trigger video path", "trigger first frame path", "nframes", "average ninstance per frame"]
        for k, v in self.ninstances_each.items():
            header.append(k)
            row.append(str(v))
        if self.meta_csvpath.exists() and self.meta_csvpath.is_file():
            with self.meta_csvpath.open(mode="a") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(row)
        else:
            with self.meta_csvpath.open(mode="w") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(header)
                csvwriter.writerow(row)

    def isOpened(self):
        return self.opened

def main(vidpath, model_detect_path, outdir, model_cls_path = None, save_instances = True, imgsz=864):

    vidname = ".".join(os.path.basename(vidpath).split(".")[:-1])

    # setting up output directory
    triggers_dir = os.path.join(outdir, "triggers")
    originals_dir = os.path.join(outdir, "originals")
    instances_dir = os.path.join(outdir, "instances")
    originals_first_frames_dir = os.path.join(originals_dir, "first_frames")
    triggers_first_frames_dir = os.path.join(triggers_dir, "first_frames")
    os.makedirs(triggers_dir, exist_ok=True)
    os.makedirs(originals_dir, exist_ok=True)
    os.makedirs(instances_dir, exist_ok=True)
    os.makedirs(originals_first_frames_dir, exist_ok=True)
    os.makedirs(triggers_first_frames_dir, exist_ok=True)

    cap = open_video(vidpath)
    #fps = cap.get(cv2.CAP_PROP_FPS)
    fps=10.0 # temporary fix
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = ultralytics.YOLO(model_detect_path, task="detect")

    print(f"""INPUT VIDEO: {vidpath}
    Resolution: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}x{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}
    FPS:        {fps}
OUTPUT DIRECTORY: {outdir}/{{originals,triggers}}
DETECTION MODEL: {model_detect_path}

Beginning processing...
""")

    subvid = None
    success = cap.isOpened()
    nframe = 0
    WAITLIMIT=5*fps # wait two seconds before closing video
    wait_counter = WAITLIMIT
    batch_size = WAITLIMIT
    while success and nframe < total_frames:

        # read frames into batch
        frames = []
        while len(frames) < batch_size:
            success, frame = cap.read()
            if success: 
                frames.append(frame)
            elif len(frames) > 0:
                frames.append(np.zeros_like(frames[0])) # loop logic ensures first frame is always present

        # inference on frames batch
        if frames: 
            batch_res = model(
                source=frames,
                #classes=[bird_class_idx],
                conf=0.46,
                iou=0.5,
                imgsz=imgsz,
                verbose=False,
                #half=True
            )
        else:
            batch_res = []
      
        for yolo_res in batch_res:
            print(f"frame {nframe+1}/{total_frames}", end=" ")
            # save bool indicating whether a bird was detected
            bird_detected = len(yolo_res.boxes.cls) > 0
            # update wait_counter if bird not detected
            wait_counter = 0 if bird_detected else wait_counter + 1
            # keep writing to video if wait limit hasn't been reached
            if wait_counter < WAITLIMIT:
                # subclassify
                if bird_detected:
                    instance_count = count_detections(yolo_res)
                    for k, v in instance_count.items():
                        print(f"{k}: {v}", end=" ")
                    print()
                    if save_instances: yolo_res.save_crop(instances_dir, add_seconds_to_timestring(vidname, nframe/fps))
                else:
                    instance_count = {}
                    print("no bird detected")
                # create video if a video isn't currently open
                if not subvid or not subvid.isOpened():
                    subvid = detected_birb_vid(cap, vidpath, nframe, outdir, WAITLIMIT, list(model.names.values()))
                
                subvid.write(yolo_res.orig_img, yolo_res.plot(conf=False), instance_count)
            else:
                print("no bird detected")
                # close the output video if it's open
                if subvid and subvid.isOpened():
                    subvid.release()
            nframe += 1
    
    cap.release() 
    if subvid and subvid.isOpened(): subvid.release() 
        
    # Closes all the frames 
    cv2.destroyAllWindows() 
       
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v")
    parser.add_argument("--model", "-m", default="yolov8n.pt")
    parser.add_argument("--output-directory", "-o", default=".")
    parser.add_argument("--cls-model", "-c", default=None)
    parser.add_argument("--save-instances", "-s", action="store_true")
    parser.add_argument("--imgsz", "-i", default=864)
    args = parser.parse_args()
    main(args.video, args.model, args.output_directory, args.cls_model, args.save_instances, args.imgsz)
